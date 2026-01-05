import yaml
import argparse
import os
import json

from easydict import EasyDict as edict


def load_yaml(path, default_path=None):
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get("inherit_from")
    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_yaml(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def dict2parser(d, pre=""):
    ret_dict = edict()
    for key, val in d.items():
        if type(val) == edict or type(val) == dict:
            ret_dict.update(dict2parser(val, pre + key + "."))
        else:
            ret_dict[pre + key] = val
    return ret_dict


def build_keys(d, keys=[], val=None):
    if len(keys) == 1:
        d[keys[0]] = val
        return
    if keys[0] not in d:
        d[keys[0]] = {}
    build_keys(d[keys[0]], keys[1:], val)


def parser2dict(d):
    ret_dict = {}

    for key, val in d.items():
        keys = key.split(".")
        build_keys(ret_dict, keys, val)
        # ret_dict[keys[-1]] = val

    return ret_dict


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


class YamlParser:
    def __init__(self, args=None, verbose=True) -> None:
        config_file_parser = argparse.ArgumentParser()
        config_file_parser.add_argument("config", type=str, help="configure file path")

        config_file_args, remaining_args = config_file_parser.parse_known_args(args=args)

        default_config_in_file = None
        if config_file_args.config is not None and os.path.exists(config_file_args.config):
            if verbose:
                print("Loading Config from %s" % config_file_args.config)
            default_config_in_file = load_yaml(config_file_args.config)

        override_config_parser = argparse.ArgumentParser()
        
        #############################
        ############ Log ############
        #############################
        override_config_parser.add_argument("--log.expname", type=str, default="test")
        override_config_parser.add_argument("--log.out_dir", type=str, default='output/tmp')
        override_config_parser.add_argument("--log.gmc_ckpt_freq", type=int, default=10000)
        override_config_parser.add_argument("--log.wandb_tags", type=lambda s: s.split(','), default=None)
        override_config_parser.add_argument('--log.log_energy_term_loss', action='store_true', default=False)

        #############################
        ########### Train ###########
        #############################
        override_config_parser.add_argument("--train.load_gmc_ckpt_path", type=str, default=None)
        override_config_parser.add_argument('--train.batch_size', type=int, default=20000)
        override_config_parser.add_argument("--train.total_iteration", type=int, default=40000)
        override_config_parser.add_argument("--train.joint_start_iteration", type=int, default=20000)

        #############################
        ####### Joint Training ######
        #############################
        override_config_parser.add_argument('--joint.turn_on_desify_prune', default=True)
        override_config_parser.add_argument('--joint.turn_on_desify', default=True)
        override_config_parser.add_argument('--joint.desify_prune_iters', type=int, default=10000)
        
        #############################
        ########### Loss ############
        #############################
        override_config_parser.add_argument("--loss.self_render_loss_weight", type=float, default=1.0)
        override_config_parser.add_argument("--loss.cross_render_loss_weight", type=float, default=1.0)
        override_config_parser.add_argument("--loss.lpips_weight", type=float, default=0.2)

        override_config_parser.add_argument("--loss.self_local_distance_weight_start", type=float, default=0.0)
        override_config_parser.add_argument("--loss.self_local_distance_weight_end", type=float, default=20.0)
        override_config_parser.add_argument("--loss.self_local_distance_weight_steps", type=int, default=20000)

        override_config_parser.add_argument("--loss.cross_local_distance_weight_start", type=float, default=0.0)
        override_config_parser.add_argument("--loss.cross_local_distance_weight_end", type=float, default=20.0)
        override_config_parser.add_argument("--loss.cross_local_distance_weight_steps", type=int, default=20000)

        override_config_parser.add_argument("--loss.local_agree_loss_neighbour_num", type=int, default=256)

        #############################
        ############ GS #############
        #############################
        override_config_parser.add_argument("--gs.checkpoint1", type=str, help="path to checkpoint file of GS1")
        override_config_parser.add_argument("--gs.checkpoint2", type=str, help="path to checkpoint file of GS2")
        override_config_parser.add_argument("--gs.pretrained_frame1_dir", type=str)
        override_config_parser.add_argument("--gs.pretrained_frame2_dir", type=str)
        override_config_parser.add_argument("--gs.load_iteration", type=int, default=30000)

        override_config_parser.add_argument("--gs.checkpoint1_gt", type=str, help="path to (pseudo) gt checkpoint file of GS1")
        override_config_parser.add_argument("--gs.checkpoint2_gt", type=str, help="path to (pseudo) gt checkpoint file of GS2")

        override_config_parser.add_argument('--gs.feat_dim', type=int, default=384)
        override_config_parser.add_argument('--gs.rp_dim', type=int, default=16)
        override_config_parser.add_argument('--gs.pca_dim', type=int, default=4)

        override_config_parser.add_argument("--gs.lr.fgs_xyz_lr", type=float, default=0.000016)
        override_config_parser.add_argument("--gs.lr.fgs_f_dc_lr", type=float, default=0.0001)
        override_config_parser.add_argument("--gs.lr.fgs_f_rest_lr", type=float, default=0.0001)
        override_config_parser.add_argument("--gs.lr.fgs_opacity_lr", type=float, default=0.05)
        override_config_parser.add_argument("--gs.lr.fgs_scaling_lr", type=float, default=0.005)
        override_config_parser.add_argument("--gs.lr.fgs_rotation_lr", type=float, default=0.0)
        override_config_parser.add_argument("--gs.lr.fgs_semantic_feature_lr", type=float, default=0.0001)

        #############################
        ######### GMC #########
        #############################
        override_config_parser.add_argument("--gmc.init_trans_zero", default=False)

        override_config_parser.add_argument("--gmc.dropout_ratio", type=float, default=0.2)
        override_config_parser.add_argument("--gmc.xyz_input_scale", type=float, default=1.0)
        override_config_parser.add_argument("--gmc.dino_input_scale", type=float, default=1.0)

        override_config_parser.add_argument("--gmc.learning_rate", type=float, default=0.0005)

        #############################
        ######### Database ##########
        #############################
        override_config_parser.add_argument("--database.color_dist_weight", type=float, default=1.0)
        override_config_parser.add_argument("--database.feat_dist_weight", type=float, default=1.0)
        override_config_parser.add_argument("--database.xyz_dist_weight", type=float, default=10.0)

        override_config_parser.add_argument('--database.noise_type', type=str, default='gumbel', choices=['gumbel', 'zero'])
        override_config_parser.add_argument("--database.noise_gumbel_mu", type=float, default=0)
        override_config_parser.add_argument("--database.noise_gumbel_beta", type=float, default=0.1)

        override_config_parser.add_argument("--database.fg_info_path", type=str, help="path to point cloud file which contains foreground info")

        #############################
        ############ Vis ############
        #############################
        override_config_parser.add_argument('--vis.load_iteration', type=int, default=40000)
        override_config_parser.add_argument('--vis.white_background', action='store_true', default=False)
        override_config_parser.add_argument('--vis.render_skip_every', type=int, default=20)

        override_config_parser.add_argument('--vis.video_view_id', type=int, default=0)

        #############################
        ############ Eval ############
        #############################
        override_config_parser.add_argument('--eval.load_iteration', type=int, default=10000)
        override_config_parser.add_argument('--eval.max_pt', type=int, default=2000)

        if default_config_in_file is not None:
            override_config_parser.set_defaults(**dict2parser(default_config_in_file))

        args = override_config_parser.parse_args(remaining_args)

        args = edict(vars(args))
        args = parser2dict(args)
        args = edict(args)
        if verbose:
            print(json.dumps(args, indent=2, ensure_ascii=False))
        self.args = args

    def print(self):
        print(json.dumps(self.args, indent=2, ensure_ascii=False))
