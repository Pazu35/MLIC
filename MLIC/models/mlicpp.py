import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder

from MLIC.MLIC.utils.func import update_registered_buffers, get_scale_table
from MLIC.MLIC.utils.ckbd import *
from MLIC.MLIC.modules.transform import *


class MLICPlusPlus(CompressionModel):

    def __init__(self, config, **kwargs):
        super().__init__(config.N, **kwargs)
        N = config.N
        M = config.M
        in_channels = getattr(config, "in_channels", 157)
        context_window = config.context_window
        num_heads_slice = 32
        slice_num = config.slice_num
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M
        if slice_ch % num_heads_slice != 0:
            num_heads_slice = 16
        assert slice_ch % num_heads_slice == 0

        self.add_sst = config.get("add_sst", False)

        if "add_seasons" in config:
            self.add_seasons = config.add_seasons
        else:
            self.add_seasons = {"use": False, "mode": None}

        if self.add_seasons["use"] and self.add_seasons["mode"] == "embed":
            self.season_embed = nn.Embedding(num_embeddings=4, embedding_dim=8)

        self.N = N
        self.M = M
        self.context_window = context_window
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        # Attention module enable/disable flags
        self.enable_channel_context = config.get("enable_channel_context", True)
        self.enable_local_context = config.get("enable_local_context", True)
        self.enable_global_inter_context = config.get("enable_global_inter_context", True)
        self.enable_global_intra_context = config.get("enable_global_intra_context", True)



        self.g_a = AnalysisTransform(N=N, M=M, in_channels=in_channels)
        self.g_s = SynthesisTransform(N=N, M=M, out_channels=in_channels)
        #self.g_s = SynthesisTransformOld(N=N, M=M, out_channels=in_channels)

        self.h_a = HyperAnalysis(M=M, N=N)
        self.h_s = HyperSynthesis(M=M, N=N)

        # Gussian Conditional
        self.gaussian_conditional = GaussianConditional(None)

        self.local_context = nn.ModuleList(
            LocalContext(dim=slice_ch)
            for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )

        # Global Reference for non-anchors
        self.global_inter_context = nn.ModuleList(
            LinearGlobalInterContext(dim=slice_ch * i, out_dim=slice_ch * 2, num_heads=slice_ch * i // num_heads_slice) if i else None  #16 #32
            for i in range(slice_num)
        )
        self.global_intra_context = nn.ModuleList(
            LinearGlobalIntraContext(dim=slice_ch) if i else None
            for i in range(slice_num)
        )



        # Calculate input dimensions based on enabled contexts
        # For anchor (idx > 0): global_inter_ctx (2*slice_ch) + channel_ctx (4*slice_ch) + hyper_params (M*2)
        anchor_ctx_dim = 0
        if self.enable_global_inter_context:
            anchor_ctx_dim += slice_ch * 2
        if self.enable_channel_context:
            anchor_ctx_dim += slice_ch * 4

        # For nonanchor (idx > 0): local_ctx (2*slice_ch) + global_intra_ctx (2*slice_ch) + 
        #                          global_inter_ctx (2*slice_ch) + channel_ctx (4*slice_ch) + hyper_params (M*2)
        nonanchor_ctx_dim = 0
        if self.enable_local_context:
            nonanchor_ctx_dim += slice_ch * 2
        if self.enable_global_intra_context:
            nonanchor_ctx_dim += slice_ch * 2
        if self.enable_global_inter_context:
            nonanchor_ctx_dim += slice_ch * 2
        if self.enable_channel_context:
            nonanchor_ctx_dim += slice_ch * 4

        # For idx == 0: only local_ctx for nonanchor
        nonanchor_ctx_dim_idx0 = (slice_ch * 2) if self.enable_local_context else 0

        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + anchor_ctx_dim, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + nonanchor_ctx_dim, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + nonanchor_ctx_dim_idx0, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )


        # self.entropy_parameters_anchor = nn.ModuleList(
        #     EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
        #     if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
        #     for i in range(slice_num)
        # )
        # self.entropy_parameters_nonanchor = nn.ModuleList(
        #     EntropyParameters(in_dim=M * 2 + slice_ch * 10, out_dim=slice_ch * 2)
        #     if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
        #     for i in range(slice_num)
        # )

        # Latent Residual Prediction
        # self.lrp_anchor = nn.ModuleList(
        #     LatentResidualPredictionOld(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
        #     for i in range(slice_num)
        # )
        # self.lrp_nonanchor = nn.ModuleList(
        #     LatentResidualPredictionOld(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
        #     for i in range(slice_num)
        # )
        
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

    def forward(self, x, season_idx=None, sst=None):
        """
        Using checkerboard context model with mask attention
        which divides y into anchor and non-anchor parts
        non-anchor use anchor as spatial context
        In addition, a channel-wise entropy model is used, too.
        Args:
            x: [B, C, H, W]
        return:
            x_hat: [B, C, H, W]
            y_likelihoods: [B, M, H // 16, W // 16]
            z_likelihoods: [B, N, H // 64, W // 64]
            likelihoods: y_likelihoods, z_likelihoods
        """

        B, C, H, W = x.shape

        if self.add_seasons["use"] and self.add_seasons["mode"] == "one_hot":
            # season_idx = x[:, 0, 0, 0].long()              # [B]
            # x = x[:, 1:, :, :]                             # remove season channel

            season_one_hot = F.one_hot(torch.tensor(season_idx,device=x.device), num_classes=4).float()  # [B, 4]
            season_one_hot = season_one_hot[:, :, None, None]              # [B, 4, 1, 1]
            season_one_hot = season_one_hot.expand(-1, -1, *x.shape[2:])   # [B, 4, H, W]

            x = torch.cat([x, season_one_hot], dim=1)      # [B, (D-1)+4, H, W]

        elif self.add_seasons["use"] and self.add_seasons["mode"] == "embed":

            season_vec = self.season_embed(torch.tensor(season_idx,device=x.device))     # [B, embed_dim]
            season_vec = season_vec[:, :, None, None]      # [B, embed_dim, 1, 1]
            season_vec = season_vec.expand(-1, -1, *x.shape[2:])  # [B, embed_dim, H, W]

            x = torch.cat([x, season_vec], dim=1)          # [B, (D-1)+embed_dim, H, W]

        if self.add_sst :
            sst = torch.tensor(sst, device=x.device, dtype=x.dtype).unsqueeze(1)                   # [B, 1, H, W]
            x = torch.cat([x, sst], dim=1)                 # [B, (D-1)+embed_dim+1, H, W]

        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
            else:
                # Gather context modules as enabled
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_global_inter_context else None
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_channel_context else None
                anchor_inputs = [hyper_params]
                if self.enable_global_inter_context:
                    anchor_inputs.insert(0, global_inter_ctx)
                if self.enable_channel_context:
                    anchor_inputs.insert(1 if self.enable_global_inter_context else 0, channel_ctx)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat(anchor_inputs, dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor) if self.enable_global_intra_context else None
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                if self.enable_global_intra_context:
                    nonanchor_inputs.insert(1 if self.enable_local_context else 0, global_intra_ctx)
                if self.enable_global_inter_context:
                    nonanchor_inputs.insert(2 if self.enable_local_context or self.enable_global_intra_context else 0, global_inter_ctx)
                if self.enable_channel_context:
                    nonanchor_inputs.insert(3 if (self.enable_local_context or self.enable_global_intra_context or self.enable_global_inter_context) else 0, channel_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)
        x_hat = x_hat[:, :C, :]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }


    def update_resolutions(self, H, W):
        for i in range(len(self.global_intra_context)):
            if i == 0:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=None)
            else:
                self.local_context[i].update_resolution(H, W, next(self.parameters()).device, mask=self.local_context[0].attn_mask)

    def compress(self, x):
        torch.cuda.synchronize()
        start_time = time.time()
        self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []


        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_global_inter_context else None
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_channel_context else None
                anchor_inputs = [hyper_params]
                if self.enable_global_inter_context:
                    anchor_inputs.insert(0, global_inter_ctx)
                if self.enable_channel_context:
                    anchor_inputs.insert(1 if self.enable_global_inter_context else 0, channel_ctx)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat(anchor_inputs, dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor) if self.enable_global_intra_context else None
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                if self.enable_global_intra_context:
                    nonanchor_inputs.insert(1 if self.enable_local_context else 0, global_intra_ctx)
                if self.enable_global_inter_context:
                    nonanchor_inputs.insert(2 if self.enable_local_context or self.enable_global_intra_context else 0, global_inter_ctx)
                if self.enable_channel_context:
                    nonanchor_inputs.insert(3 if (self.enable_local_context or self.enable_global_intra_context or self.enable_global_inter_context) else 0, channel_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time
        }

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        start_time = time.time()
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)


        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)
            else:
                # Anchor
                global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_global_inter_context else None
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1)) if self.enable_channel_context else None
                anchor_inputs = [hyper_params]
                if self.enable_global_inter_context:
                    anchor_inputs.insert(0, global_inter_ctx)
                if self.enable_channel_context:
                    anchor_inputs.insert(1 if self.enable_global_inter_context else 0, channel_ctx)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat(anchor_inputs, dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor) if self.enable_global_intra_context else None
                local_ctx = self.local_context[idx](slice_anchor) if self.enable_local_context else None
                nonanchor_inputs = [hyper_params]
                if self.enable_local_context:
                    nonanchor_inputs.insert(0, local_ctx)
                if self.enable_global_intra_context:
                    nonanchor_inputs.insert(1 if self.enable_local_context else 0, global_intra_ctx)
                if self.enable_global_inter_context:
                    nonanchor_inputs.insert(2 if self.enable_local_context or self.enable_global_intra_context else 0, global_inter_ctx)
                if self.enable_channel_context:
                    nonanchor_inputs.insert(3 if (self.enable_local_context or self.enable_global_intra_context or self.enable_global_inter_context) else 0, channel_ctx)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat(nonanchor_inputs, dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        torch.cuda.synchronize()
        end_time = time.time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
