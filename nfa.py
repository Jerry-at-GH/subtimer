# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
from pathlib import Path
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import resample

import torch
from omegaconf import OmegaConf
from nfa_utils.data_prep import (
    add_t_start_end_to_utt_obj,
    get_batch_starts_ends,
    get_batch_variables,
    get_manifest_lines_batch,
    is_entry_in_all_lines,
)
from nfa_utils.make_ctm_files import make_ctm_files
from nfa_utils.make_output_manifest import write_manifest_out_line
from nfa_utils.viterbi_decoding import viterbi_decoding

from vad import ensemble_from_dir

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Align the utterances in manifest_filepath. 
Results are saved in ctm files in output_dir.

Arguments:
    pretrained_name: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded
        from NGC and used for generating the log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
    model_path: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the
        log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
        Note: if a model_path is provided, it will override the pretrained_name.
    manifest_filepath: filepath to the manifest of the data you want to align,
        containing 'audio_filepath' and 'text' fields.
    output_dir: the folder where output CTM files and new JSON manifest will be saved.
    align_using_pred_text: if True, will transcribe the audio using the specified model and then use that transcription 
        as the reference text for the forced alignment. 
    transcribe_device: None, or a string specifying the device that will be used for generating log-probs (i.e. "transcribing").
        The string needs to be in a format recognized by torch.device().
    viterbi_device: None, or string specifying the device that will be used for doing Viterbi decoding. 
        The string needs to be in a format recognized by torch.device().
    batch_size: int specifying batch size that will be used for generating log-probs and doing Viterbi decoding.
    use_local_attention: boolean flag specifying whether to try to use local attention for the ASR Model (will only
        work if the ASR Model is a Conformer model). If local attention is used, we will set the local attention context 
        size to [64,64].
    additional_segment_grouping_separator: an optional string used to separate the text into smaller segments. 
        If this is not specified, then the whole text will be treated as a single segment. 
    remove_blank_tokens_from_ctm:  a boolean denoting whether to remove <blank> tokens from token-level output CTMs. 
    audio_filepath_parts_in_utt_id: int specifying how many of the 'parts' of the audio_filepath
        we will use (starting from the final part of the audio_filepath) to determine the 
        utt_id that will be used in the CTM files. Note also that any spaces that are present in the audio_filepath 
        will be replaced with dashes, so as not to change the number of space-separated elements in the 
        CTM files.
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 1 => utt_id will be "e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 2 => utt_id will be "d_e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 3 => utt_id will be "c_d_e1"
    use_buffered_infer: False, if set True, using streaming to do get the logits for alignment
                        This flag is useful when aligning large audio file.
                        However, currently the chunk streaming inference does not support batch inference,
                        which means even you set batch_size > 1, it will only infer one by one instead of doing
                        the whole batch inference together.
    chunk_len_in_secs: float chunk length in seconds
    total_buffer_in_secs: float  Length of buffer (chunk + left and right padding) in seconds
    chunk_batch_size: int batch size for buffered chunk inference,
                      which will cut one audio into segments and do inference on chunk_batch_size segments at a time

    simulate_cache_aware_streaming: False, if set True, using cache aware streaming to do get the logits for alignment

    save_output_file_formats: List of strings specifying what type of output files to save (default: ["ctm"])
    ctm_file_config: CTMFileConfig to specify the configuration of the output CTM files
"""


@dataclass
class CTMFileConfig:
    remove_blank_tokens: bool = False
    # minimum duration (in seconds) for timestamps in the CTM.If any line in the CTM has a
    # duration lower than this, it will be enlarged from the middle outwards until it
    # meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file.
    # Note that this may cause timestamps to overlap.
    minimum_timestamp_duration: float = 0


@dataclass
class AlignmentConfig:
    # Required configs
    pretrained_name: Optional[str] = "nvidia/parakeet-tdt_ctc-0.6b-ja"
    model_path: Optional[str] = None
    manifest_filepath: Optional[str] = None
    output_dir: Optional[str] = None

    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = "cpu"
    viterbi_device: Optional[str] = "cpu"  # GPU VRAM is usually not enough if not chunking
    batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[str] = None
    audio_filepath_parts_in_utt_id: int = 1

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False  # chunking would lead to very poor performance

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False

    # Output file configs
    save_output_file_formats: List[str] = field(default_factory=lambda: ["ctm"])
    ctm_file_config: CTMFileConfig = field(default_factory=lambda: CTMFileConfig())

    # VAD prior integration
    use_vad_prior: bool = False
    vad_prior_k: float = 1.0  # strength of prior; 0 disables even if flag set
    vad_dir_name: str = "vad"  # subfolder next to audio containing *.npy
    vad_min_prob: float = 1e-6  # avoid log(0)


@hydra_runner(config_name="AlignmentConfig", schema=AlignmentConfig)
def main(cfg: AlignmentConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # Validation
    assert cfg.pretrained_name or cfg.model_path, "Need either pretrained_name or model_path"
    assert cfg.manifest_filepath and cfg.output_dir, "Need manifest_filepath and output_dir"
    assert cfg.batch_size >= 1 and cfg.ctm_file_config.minimum_timestamp_duration >= 0
    if not cfg.align_using_pred_text:
        assert is_entry_in_all_lines(cfg.manifest_filepath, "text"), "Manifest missing 'text' entries"

    # init devices
    transcribe_device = torch.device(cfg.transcribe_device)
    viterbi_device = torch.device(cfg.viterbi_device)

    # load model
    model, _ = setup_model(cfg, transcribe_device)
    model.eval()

    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoder_type="ctc")

    if cfg.use_local_attention:
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])

    assert isinstance(model, (EncDecCTCModel, EncDecHybridRNNTCTCModel)), "Only CTC models supported"

    buffered_chunk_params = {}
    if cfg.use_buffered_chunked_streaming:
        model_cfg = copy.deepcopy(model._cfg)

        OmegaConf.set_struct(model_cfg.preprocessor, False)
        # some changes for streaming scenario
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0

        if model_cfg.preprocessor.normalize != "per_feature":
            logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")
        # Disable config overwriting
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        feature_stride = model_cfg.preprocessor["window_stride"]
        model_stride_in_secs = feature_stride * cfg.model_downsample_factor
        total_buffer = cfg.total_buffer_in_secs
        chunk_len = float(cfg.chunk_len_in_secs)
        tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
        mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
        logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

        model = FrameBatchASR(
            asr_model=model,
            frame_len=chunk_len,
            total_buffer=cfg.total_buffer_in_secs,
            batch_size=cfg.chunk_batch_size,
        )
        buffered_chunk_params = {
            "delay": mid_delay,
            "model_stride_in_secs": model_stride_in_secs,
            "tokens_per_chunk": tokens_per_chunk,
        }
    # get start and end line IDs of batches
    starts, ends = get_batch_starts_ends(cfg.manifest_filepath, cfg.batch_size)

    # init output_timestep_duration = None and we will calculate and update it during the first batch
    output_timestep_duration = None

    # init f_manifest_out
    os.makedirs(cfg.output_dir, exist_ok=True)
    tgt_manifest_name = str(Path(cfg.manifest_filepath).stem) + "_with_output_file_paths.json"
    tgt_manifest_filepath = str(Path(cfg.output_dir) / tgt_manifest_name)
    f_manifest_out = open(tgt_manifest_filepath, "w")

    # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        manifest_lines_batch = get_manifest_lines_batch(cfg.manifest_filepath, start, end)

        (log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration) = get_batch_variables(
            manifest_lines_batch,
            model,
            cfg.additional_segment_grouping_separator,
            cfg.align_using_pred_text,
            cfg.audio_filepath_parts_in_utt_id,
            output_timestep_duration,
            cfg.simulate_cache_aware_streaming,
            cfg.use_buffered_chunked_streaming,
            buffered_chunk_params,
        )

        # apply VAD prior to log_probs before Viterbi
        if cfg.use_vad_prior and cfg.vad_prior_k > 0:
            if hasattr(model, "tokenizer"):
                BLANK_ID = model.blank_id if hasattr(model, "blank_id") else len(model.tokenizer.vocab)
            else:
                BLANK_ID = len(model.decoder.vocabulary)
            eps = float(cfg.vad_min_prob)
            k_prior = float(cfg.vad_prior_k)
            for b, utt_obj in enumerate(utt_obj_batch):
                T_b = int(T_batch[b])
                audio_fp = utt_obj.audio_filepath
                vad_dir = os.path.join(os.path.dirname(audio_fp) or ".", cfg.vad_dir_name)
                vad_pred = ensemble_from_dir(vad_dir)

                # resample to T_b
                vad_res = np.clip(resample(vad_pred, T_b), 0.0, 1.0)
                p = np.clip(vad_res, eps, 1.0 - eps)
                nonblank_boost = np.log(p**k_prior)  # shape (T_b,)
                blank_boost = np.log((1.0 - p) ** k_prior)

                # convert to torch
                nb = torch.from_numpy(nonblank_boost).float()
                bb = torch.from_numpy(blank_boost).float()

                # add nonblank_boost to all tokens
                log_probs_batch[b, :T_b, :] = log_probs_batch[b, :T_b, :] + nb.unsqueeze(1)
                # adjust blank column to achieve blank_boost
                delta_blank = (bb - nb).unsqueeze(1)  # (T_b,1)
                log_probs_batch[b, :T_b, BLANK_ID] = log_probs_batch[b, :T_b, BLANK_ID] + delta_blank.squeeze(1)

        alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device)

        for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):
            utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)

            if "ctm" in cfg.save_output_file_formats:
                utt_obj = make_ctm_files(
                    utt_obj,
                    cfg.output_dir,
                    cfg.ctm_file_config,
                )

            write_manifest_out_line(
                f_manifest_out,
                utt_obj,
            )

    f_manifest_out.close()
    return None


if __name__ == "__main__":
    main()
