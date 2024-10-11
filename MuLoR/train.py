import os
import faulthandler
faulthandler.enable()
import argparse
import glob
import logging
import warnings

warnings.filterwarnings("ignore")

import random
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from feature_adj_utils import load_and_cache_examples, processors
from tokenization_dagn import arg_tokenizer
from MuLoR import MuLoR

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_model(train_preds, train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc, global_step,
                   test=False, save_result=False):
    train_preds = np.argmax(train_preds, axis=1)
    train_acc = simple_accuracy(train_preds, train_label_ids)
    train_preds = None
    train_label_ids = None
    results = evaluate(args, model, tokenizer, test, save_result)
    logger.info(
        "train acc: %s, dev acc: %s, loss: %s, global steps: %s",
        str(train_acc),
        str(results["eval_acc"]),
        str(results["eval_loss"]),
        str(global_step),
    )
    tb_writer.add_scalar("training/acc", train_acc, global_step)
    for key, value in results.items():
        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
    if results["eval_acc"] > best_dev_acc:
        best_dev_acc = results["eval_acc"]
        best_steps = global_step
        logger.info("achieve BEST dev acc: %s at global step: %s",
                    str(best_dev_acc),
                    str(best_steps)
                    )
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # 处理分布式/并行训练
        # 修改保存逻辑，保存为 .pt 文件
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
        # 保存 tokenizer 的词汇表
        tokenizer.save_vocabulary(output_dir)
        # 保存其他必要的文件，例如 tokenizer 的配置
        tokenizer.save_pretrained(output_dir)
        # 保存训练参数
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
        with open(txt_dir, 'w') as f:
            rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
            f.write(rs)
            tb_writer.add_text('best_results', rs, global_step)

    return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

def save_model(args, model, tokenizer, global_step):
    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # 处理分布式/并行训练
    # 修改保存逻辑，保存为 .pt 文件
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
    # 保存 tokenizer 的词汇表
    tokenizer.save_vocabulary(output_dir)
    # 保存其他必要的文件，例如 tokenizer 的配置
    tokenizer.save_pretrained(output_dir)
    # 保存训练参数
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

def train(args, train_dataset, model, tokenizer):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_log_dir = os.path.join('summaries', os.path.basename(args.output_dir))
        tb_writer = SummaryWriter(tb_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.number_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    exec('args.adam_betas = ' + args.adam_betas)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=args.adam_betas,
        eps=args.adam_epsilon,
    )
    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), \
        "--only can set one of --warmup_steps and --warmup_proportion"

    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (
            torch.distributed.get_world_size()
            if args.local_rank != -1
            else 1
        ),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None
    model.zero_grad()
    set_seed(args)

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                argument_bpe_ids=batch[4],
                punct_bpe_ids=batch[6],
                keytokensids=batch[7],
                keymask=batch[8],
                key_segid=batch[11],
                SVO_ids=batch[9],
                SVO_mask=batch[10],
                adj_SVO=batch[13],
                labels=batch[12],
                passage_mask=batch[2],
                question_mask=batch[3],
            )
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if args.n_gpu == 1:
                if train_preds is None:
                    train_preds = logits.detach().cpu().numpy()
                    train_label_ids = batch[12].detach().cpu().numpy()
                else:
                    train_preds = np.append(train_preds, logits.detach().cpu().numpy(), axis=0)
                    train_label_ids = np.append(
                        train_label_ids, batch[12].detach().cpu().numpy(), axis=0
                    )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    if args.local_rank == -1 and args.evaluate_during_training:
                        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds,
                                                                                                           train_label_ids,
                                                                                                           tb_writer,
                                                                                                           args, model,
                                                                                                           tokenizer,
                                                                                                           best_steps,
                                                                                                           best_dev_acc,
                                                                                                           global_step)
                    tb_writer.add_scalar("training/lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "training/loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            del batch, outputs, loss
            torch.cuda.empty_cache()

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps

def evaluate(args, model, tokenizer, test=False, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, tokenizer, arg_tokenizer, evaluate=not test, test=test
        )
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=args.number_workers,
            pin_memory=torch.cuda.is_available(),
        )

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                outputs = model(
                    input_ids=batch[0],
                    attention_mask=batch[1],
                    argument_bpe_ids=batch[4],
                    punct_bpe_ids=batch[6],
                    keytokensids=batch[7],
                    keymask=batch[8],
                    key_segid=batch[11],
                    SVO_ids=batch[9],
                    SVO_mask=batch[10],
                    adj_SVO=batch[13],
                    labels=batch[12],
                    passage_mask=batch[2],
                    question_mask=batch[3],
                )
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[12].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, batch[12].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(
            eval_output_dir, f"eval_results_{'test' if test else 'dev'}.txt"
        )

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_model(args, checkpoint_path, max_rel_id, feature_dim_list):
    # 初始化模型结构
    model = MuLoR(
        args.model_name_or_path,
        max_rel_id=max_rel_id,
        feature_dim_list=feature_dim_list,
        device=args.device,
    )
    # 加载模型的状态字典
    model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
    model.to(args.device)
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
        help="The input data dir. Should contain the data files for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--task_name",
        default="reclor",
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        '--adam_betas', default='(0.9, 0.98)', type=str, help='Betas for Adam optimizer'
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--no_clip_grad_norm", action="store_true", help="Whether not to clip grad norm"
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs", default=15, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup ratios."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=200, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=800, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--number_workers", type=int, default=4, help="Number of workers for data loading"
    )

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            " Use --overwrite_output_dir to overcome."
        )

    # Setup distant debugging if needed
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError(f"Task not found: {args.task_name}")

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config = RobertaConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-model",
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)

    max_rel_id = int(max(relations.values()))
    feature_dim_list = [config.hidden_size] * 2

    model = MuLoR(
        args.model_name_or_path,
        max_rel_id=max_rel_id,
        feature_dim_list=feature_dim_list,
        device=args.device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Number of parameters: %.2fM", total_params / 1e6)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, arg_tokenizer, evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    results = {}
    print(args.do_eval)
    if args.local_rank in [-1, 0]:
        print("sss")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if "checkpoint" in checkpoint else ""

            # 构建模型并加载状态字典
            checkpoint_path = os.path.join(checkpoint, 'model.pt')
            model = load_model(args, checkpoint_path, max_rel_id, feature_dim_list)
            result = evaluate(args, model, tokenizer, prefix=prefix,test=True)
            result = {f"{k}_{global_step}": v for k, v in result.items()}
            results.update(result)

    # 如果需要，可以在这里处理 `do_test`

if __name__ == "__main__":
    main()
