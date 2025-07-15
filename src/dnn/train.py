from comet_ml import Experiment
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import os
import yaml
import numpy as np
import librosa

import torch
import torch.nn as nn
from torch.utils import data

from cnn_gru import spec_CNN_GRU


def balance_classes(lines_small, lines_big, np_seed):
    """
    Balance number of sample per class.
    Designed for Binary(two-class) classification.
    """

    len_small_lines = len(lines_small)
    len_big_lines = len(lines_big)

    np.random.seed(np_seed)
    np.random.shuffle(lines_big)
    new_lines = lines_small + lines_big[:len_small_lines]
    np.random.shuffle(new_lines)

    return new_lines


def get_utt_list(src_dir):
    """
    Get all .npy files from the unified data directory
    """
    l_utt = []
    for f in os.listdir(src_dir):
        if f.endswith(".npy"):
            l_utt.append(f.split(".")[0])
    return l_utt


def split_genSpoof_unified(l_utt, protocol_files, return_dic_meta=False):
    """
    Split genuine and spoofed samples using multiple protocol files
    protocol_files should be a list of protocol file paths
    """
    l_gen, l_spo = [], []
    d_meta = {}

    # Read all protocol files and combine them
    for protocol_file in protocol_files:
        if os.path.exists(protocol_file):
            with open(protocol_file, "r") as f:
                l_meta = f.readlines()
            for line in l_meta:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    key = parts[0].split(".")[0]
                    label = parts[1]
                    d_meta[key] = 1 if label == "genuine" else 0

    # Split into genuine and spoofed lists
    for key in l_utt:
        if key in d_meta:
            if d_meta[key] == 1:
                l_gen.append(key)
            else:
                l_spo.append(key)
        else:
            # If not found in protocol, you may want to handle this case
            print(f"Warning: {key} not found in protocol files")

    if return_dic_meta:
        return l_gen, l_spo, d_meta
    else:
        return l_gen, l_spo


class Dataset_ASVspoof_Unified(data.Dataset):
    def __init__(self, list_IDs, labels, nb_time, base_dir):
        """
        Unified dataset class for single directory
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.nb_time = nb_time
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        file_path = os.path.join(self.base_dir, ID + ".npy")
        X = np.load(file_path)

        nb_time = X.shape[1]
        if nb_time > self.nb_time:
            start_idx = np.random.randint(low=0, high=nb_time - self.nb_time)
            X = X[:, start_idx : start_idx + self.nb_time, :]
        elif nb_time < self.nb_time:
            nb_dup = int(self.nb_time / nb_time) + 1
            X = np.tile(X, (1, nb_dup, 1))[:, : self.nb_time, :]

        y = self.labels[ID]

        return X, y


def save_model_config(save_dir, parser):
    """
    Save model configuration for later use in prediction
    """
    config_path = os.path.join(save_dir, "model_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(parser, f)
    print(f"Model configuration saved to {config_path}")


if __name__ == "__main__":
    # load yaml file & set comet_ml config
    _abspath = os.path.abspath(__file__)
    dir_yaml = os.path.splitext(_abspath)[0] + ".yaml"
    with open(dir_yaml, "r") as f_yaml:
        parser = yaml.load(f_yaml, Loader=yaml.FullLoader)

    experiment = Experiment(
        api_key="YOUR API KEY",
        project_name="YOUR_PROJECT_NAME",
        workspace="YOUR_WORKSPACE_NAME",
        disabled=bool(parser["comet_disable"]),
    )
    experiment.set_name(parser["name"])

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % parser["gpu_idx"][0] if cuda else "cpu")

    # Set unified data directory
    unified_data_dir = "./data/spec_magnitude_2048_800_480"

    # Get all utterances from the unified directory
    l_all_utt = get_utt_list(unified_data_dir)
    print(f"Found {len(l_all_utt)} .npy files in {unified_data_dir}")

    # Prepare protocol file paths
    protocol_files = []
    if "dir_meta_trn" in parser and "DB" in parser:
        protocol_files.append(parser["DB"] + parser["dir_meta_trn"])
    if "dir_meta_dev" in parser and "DB" in parser:
        protocol_files.append(parser["DB"] + parser["dir_meta_dev"])
    if "dir_meta_eval" in parser and "DB" in parser:
        protocol_files.append(parser["DB"] + parser["dir_meta_eval"])

    # If no protocol files specified, create default paths
    if not protocol_files:
        protocol_files = ["./data/train.txt", "./data/dev.txt", "./data/eval.txt"]

    print(f"Using protocol files: {protocol_files}")

    # Split genuine and spoofed samples using unified approach
    l_gen_all, l_spo_all, d_label_all = split_genSpoof_unified(
        l_utt=l_all_utt, protocol_files=protocol_files, return_dic_meta=True
    )

    print(
        f"Training on ALL data: {len(l_gen_all)} genuine and {len(l_spo_all)} spoofed samples"
    )

    # Create a small validation set (10%) for monitoring training progress
    np.random.seed(42)

    # Take small portion for validation monitoring
    val_size = 0.1
    n_gen_val = max(1, int(val_size * len(l_gen_all)))
    n_spo_val = max(1, int(val_size * len(l_spo_all)))

    np.random.shuffle(l_gen_all)
    np.random.shuffle(l_spo_all)

    l_gen_val = l_gen_all[:n_gen_val]
    l_gen_train = l_gen_all[n_gen_val:]

    l_spo_val = l_spo_all[:n_spo_val]
    l_spo_train = l_spo_all[n_spo_val:]

    print(f"Training set: {len(l_gen_train)} genuine, {len(l_spo_train)} spoofed")
    print(f"Validation set: {len(l_gen_val)} genuine, {len(l_spo_val)} spoofed")

    # get balanced validation utterance list
    l_val_utt = balance_classes(l_gen_val, l_spo_val, 0)

    # define validation dataset generator
    valset = Dataset_ASVspoof_Unified(
        list_IDs=l_val_utt,
        labels=d_label_all,
        nb_time=parser["nb_time"],
        base_dir=unified_data_dir,
    )
    valset_gen = data.DataLoader(
        valset,
        batch_size=parser["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=parser["nb_proc_db"],
    )

    # set save directory
    save_dir = parser["save_dir"] + parser["name"] + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + "results/"):
        os.makedirs(save_dir + "results/")
    if not os.path.exists(save_dir + "models/"):
        os.makedirs(save_dir + "models/")

    # Save model configuration for later prediction
    save_model_config(save_dir, parser)

    # log experiment parameters
    f_params = open(save_dir + "f_params.txt", "w")
    for k, v in parser.items():
        print(k, v)
        f_params.write("{}:\t{}\n".format(k, v))
    f_params.write("DNN model params\n")

    for k, v in parser["model"].items():
        f_params.write("{}:\t{}\n".format(k, v))
    f_params.close()

    if not bool(parser["comet_disable"]):
        experiment.log_parameters(parser)
        experiment.log_parameters(parser["model"])

    # define model
    model = spec_CNN_GRU(parser["model"], device=device, do_pretrn=True).to(device)

    # log model summary
    with open(save_dir + "summary.txt", "w+") as f_summary:
        model.summary(
            input_size=(
                parser["model"]["in_channels"],
                parser["nb_time"],
                parser["feat_dim"],
            ),
            print_fn=lambda x: f_summary.write(x + "\n"),
        )
        model.forward_mode = "gru"
        model.summary(
            input_size=(
                parser["model"]["in_channels"],
                parser["nb_time"],
                parser["feat_dim"],
            ),
            print_fn=lambda x: f_summary.write(x + "\n"),
        )
        model.forward_mode = "cnn"

    if len(parser["gpu_idx"]) > 1:
        model = nn.DataParallel(model, device_ids=parser["gpu_idx"])

    # set objective function
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    params = list(model.parameters())
    if parser["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=parser["lr"],
            momentum=parser["opt_mom"],
            weight_decay=parser["wd"],
            nesterov=bool(parser["nesterov"]),
        )
    elif parser["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=parser["lr"],
            weight_decay=parser["wd"],
            amsgrad=bool(parser["amsgrad"]),
        )

    if bool(parser["do_lr_dec"]):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=parser["lrdec_milestones"], gamma=parser["lrdec"]
        )

    ##########################################
    # train/val################################
    ##########################################
    assert model.forward_mode == "cnn"
    best_eer = 99.0
    f_eer = open(save_dir + "eers.txt", "a", buffering=1)

    for epoch in tqdm(range(parser["epoch"])):
        if epoch == int(parser["pretrn_epoch"]):
            model.forward_mode = "gru"
            model.load_state_dict(torch.load(save_dir + "models/best.pt"))
            f_eer.write("START GRU\n")

        # make classwise-balanced utt list for this epoch using ALL training data
        trn_list_cur = balance_classes(l_gen_train, l_spo_train, int(epoch))

        # define dataset generators using unified directory
        trnset = Dataset_ASVspoof_Unified(
            list_IDs=trn_list_cur,
            labels=d_label_all,
            nb_time=parser["nb_time"],
            base_dir=unified_data_dir,
        )
        trnset_gen = data.DataLoader(
            trnset,
            batch_size=parser["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=parser["nb_proc_db"],
        )

        # train phase
        model.train()
        with tqdm(total=len(trnset_gen), ncols=70) as pbar:
            for m_batch, m_label in trnset_gen:
                m_batch, m_label = m_batch.to(device), m_label.to(device)

                _, output = model(m_batch)
                loss = criterion(output, m_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description("epoch%d,loss:%.3f" % (epoch, loss))
                pbar.update(1)

        if not bool(parser["comet_disable"]):
            experiment.log_metric("loss", loss)

        # validation phase (for monitoring only)
        if bool(parser["do_lr_dec"]):
            lr_scheduler.step()

        model.eval()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(valset_gen), ncols=70) as pbar:
                y_score = []
                y = []
                for m_batch, m_label in valset_gen:
                    m_batch = m_batch.to(device)
                    y.extend(list(m_label))
                    code, out = model(m_batch)
                    y_score.extend(out.cpu().numpy()[:, 0])
                    pbar.update(1)

            # calculate EER
            f_res = open(save_dir + "results/epoch%s.txt" % (epoch), "w")
            for _s, _t in zip(y, y_score):
                f_res.write("{score} {target}\n".format(score=_s, target=_t))
            f_res.close()

            fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=0)
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            print(f"Epoch {epoch}, Validation EER: {eer:.4f}")

            if not bool(parser["comet_disable"]):
                experiment.log_metric("val_eer", eer)
            f_eer.write("%d %f \n" % (epoch, eer))

            # record best validation model
            if float(eer) < best_eer:
                print("New best EER: %f" % float(eer))
                best_eer = float(eer)
                if not bool(parser["comet_disable"]):
                    experiment.log_metric("best_val_eer", eer)

                # save best model
                if len(parser["gpu_idx"]) > 1:
                    torch.save(model.module.state_dict(), save_dir + "models/best.pt")
                else:
                    torch.save(model.state_dict(), save_dir + "models/best.pt")

            if not bool(parser["save_best_only"]):
                # save model
                if len(parser["gpu_idx"]) > 1:
                    torch.save(
                        model.module.state_dict(),
                        save_dir + "models/%d-%.6f.pt" % (epoch, eer),
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        save_dir + "models/%d-%.6f.pt" % (epoch, eer),
                    )

    f_eer.close()

    # Save final model
    final_model_path = save_dir + "models/final_model.pt"
    if len(parser["gpu_idx"]) > 1:
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)

    print(f"Training completed!")
    print(f"Best model saved at: {save_dir}models/best.pt")
    print(f"Final model saved at: {final_model_path}")
    print(f"Model config saved at: {save_dir}model_config.yaml")
