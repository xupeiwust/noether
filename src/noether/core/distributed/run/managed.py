#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import platform

from torch.distributed import barrier, destroy_process_group, init_process_group

from noether.core.distributed.config import (
    get_local_rank,
    get_managed_rank,
    get_managed_world_size,
    get_num_nodes,
    is_managed,
)
from noether.core.distributed.utils import accelerator_to_device, check_single_device_visible, get_backend

logger = logging.getLogger(__name__)


def run_managed(main, accelerator="gpu", devices=None):
    assert is_managed()
    # some HPCs dont set CUDA_VISIBLE_DEVICES at all (e.g. lux)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        local_rank = get_local_rank()
        visible_device_str = str(local_rank)
        logger.info(
            f"no CUDA_VISIBLE_DEVICES found -> set CUDA_VISIBLE_DEVICES={visible_device_str} (local_rank={local_rank})"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device_str
    else:
        # srun doesnt set correct CUDA_VISIBLE_DEVICES
        split = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(split) > 1:
            local_rank = get_local_rank()
            visible_device_str = split[local_rank]
            logger.info(
                f"found multiple visible devices (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}) "
                f"-> set CUDA_VISIBLE_DEVICES={visible_device_str} (local_rank={local_rank})"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device_str
    check_single_device_visible(accelerator=accelerator)
    assert devices is None, f"devices are set implicitly via environment (devices should be None but is '{devices}')"
    world_size = get_managed_world_size()
    if world_size == 1:
        # no need for setting up distributed stuff
        _run_managed_singleprocess(accelerator, main)
    else:
        # use all GPUs for training
        _run_managed_multiprocess(accelerator, main)


def _run_managed_singleprocess(accelerator, main):
    # single process
    logger.info("running single process slurm training")
    device = accelerator_to_device(accelerator)
    main(device=device)


def _run_managed_multiprocess(accelerator, main):
    # setup MASTER_ADDR & MASTER_PORT
    if not os.environ.get("MASTER_ADDR", ""):
        if "SLURM_JOB_NODELIST" in os.environ:
            os.environ["MASTER_ADDR"] = os.environ["SLURM_JOB_NODELIST"].split(",")[0]
        else:
            raise RuntimeError("SLURM_JOB_NODELIST not found in environment, cannot set MASTER_ADDR")
    if not os.environ.get("MASTER_PORT", ""):
        if "SLURM_JOB_ID" in os.environ:
            # derive a port from the slurm job id
            slurm_job_id = int(os.environ["SLURM_JOB_ID"])
            master_port = 15000 + (slurm_job_id % 10000)
            os.environ["MASTER_PORT"] = str(master_port)
            logger.info(f"setting MASTER_PORT={master_port} derived from SLURM_JOB_ID={slurm_job_id}")
        else:
            raise RuntimeError("SLURM_JOB_ID not found in environment, cannot set MASTER_PORT")
    # get config from env variables
    world_size = get_managed_world_size()
    rank = get_managed_rank()

    # init process group
    logger.info(
        f"initializing rank={rank} local_rank={get_local_rank()} "
        f"nodes={get_num_nodes()} hostname={platform.uname().node} "
        f"master_addr={os.environ['MASTER_ADDR']} master_port={os.environ['MASTER_PORT']} "
        f"(waiting for all {world_size} processes to connect)"
    )
    init_process_group(backend=get_backend(accelerator), init_method="env://", world_size=world_size, rank=rank)
    barrier()

    # start main_single
    device = accelerator_to_device(accelerator)
    main(device=device)
    # try:
    #     main(device=device)
    # except:
    #     # sometimes NCCL will time-out but not kill all processes -> job stays alive and idles GPUs
    #     # this is hard to test, because the actual case where this happens is quite rare so its not tested thoroughly
    #     import traceback
    #     traceback.print_exc()
    #     if "SLURM_JOB_ID" in os.environ:
    #         os.system(f"scancel {os.environ['SLURM_JOB_ID']}")
    #     else:
    #         raise NotImplementedError
    #     raise

    destroy_process_group()
