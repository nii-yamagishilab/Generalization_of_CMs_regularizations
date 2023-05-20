import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pytorch_adapt.layers import GradientReversal
sys.path.insert(0, '/home/smg/zengchang/code/thesis/chapter2/code')

import lcnn_lfcc_meta_learn_grl.model.feature as front_end
from lcnn_lfcc_meta_learn_grl.meta_modules import modules

from ipdb import set_trace

class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D) 
    
    MaxFeatureMap2D(max_dim=1)
    
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """
    def __init__(self, max_dim = 1):
        super().__init__()
        self.max_dim = max_dim
        
    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)
        
        shape = list(inputs.size())
        
        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim]//2
        shape.insert(self.max_dim, 2)
        
        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m

class DomainAlignment(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, domain_align_weight):
        super(DomainAlignment, self).__init__()
        self.grl = GradientReversal(weight = domain_align_weight)
        self.input_layer = nn.Sequential(
                modules.Linear_fw(in_dim, hidden_dim),
                nn.ReLU(),
                modules.BatchNorm1d_fw(hidden_dim)
                )
        self.hidden_layers = nn.Sequential(
                modules.Linear_fw(hidden_dim, hidden_dim),
                nn.ReLU(),
                modules.BatchNorm1d_fw(hidden_dim)
                )
        self.output_layer = modules.Linear_fw(hidden_dim, out_dim)

    def forward(self, embedding):
        x = self.grl(embedding)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class LCNN(nn.Module):
    def __init__(self, in_dim, out_dim, domain_align_weight = 1.0, domain_num = 10, args = None, mean_std = None):
        super(LCNN, self).__init__()
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim, out_dim, args, mean_std)
        self.input_mean = in_m #nn.Parameter(in_m, requires_grad=False)
        self.input_std = in_s #nn.Parameter(in_s, requires_grad=False)
        self.output_mean = out_m #nn.Parameter(out_m, requires_grad=False)
        self.output_std = out_s #nn.Parameter(out_s, requires_grad=False)
        
        # a flag for debugging (by default False)
        self.model_debug = False
        self.flag_validation = False
        #####
        
        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000
        
        # flag for balanced class (temporary use)
        self.v_flag = 1

        ####
        # front-end configuration
        #  multiple front-end configurations may be used
        #  by default, use a single front-end
        ####    
        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # 750 frames are quite long for ASVspoof2019 LA with frame_shift = 10ms
        self.v_truncate_lens = [10 * 16 * 750 // x for x in self.frame_hops]

        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = 1
        self.alignment_dim = domain_num

        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 2nd part of the classifier
        self.m_output_act = []
        self.m_output = []
        self.alignment = []
        # front-end
        self.m_frontend = []

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end
        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                nn.Sequential(
                    modules.Conv2d_fw(1, 64, [5, 5], 1, padding=[2, 2]),
                    MaxFeatureMap2D(),
                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    modules.Conv2d_fw(32, 64, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    nn.BatchNorm2d(32, affine=False),
                    modules.Conv2d_fw(32, 96, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),
                    nn.BatchNorm2d(48, affine=False),

                    modules.Conv2d_fw(48, 96, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    nn.BatchNorm2d(48, affine=False),
                    modules.Conv2d_fw(48, 128, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    modules.Conv2d_fw(64, 128, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    nn.BatchNorm2d(64, affine=False),
                    modules.Conv2d_fw(64, 64, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),
                    nn.BatchNorm2d(32, affine=False),

                    modules.Conv2d_fw(32, 64, [1, 1], 1, padding=[0, 0]),
                    MaxFeatureMap2D(),
                    nn.BatchNorm2d(32, affine=False),
                    modules.Conv2d_fw(32, 64, [3, 3], 1, padding=[1, 1]),
                    MaxFeatureMap2D(),
                    nn.MaxPool2d([2, 2], [2, 2])
                )
            )
            self.m_output_act.append(
                nn.Sequential(
                    nn.Dropout(0.7),
                    modules.Linear_fw((trunc_len // 16) * 
                                    (lfcc_dim // 16) * 32, 160),
                    MaxFeatureMap2D()
                    #  modules.Linear_fw(80, self.v_emd_dim)
                )
            )
            self.m_output.append(
                nn.Sequential(
                    modules.Linear_fw(80, self.v_emd_dim)
                )
            )

            self.alignment.append(
                nn.Sequential(
                    DomainAlignment(80, domain_num, 128, domain_align_weight)
                )
            )
            
            self.m_frontend.append(
                front_end.LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   with_energy=True)
            )

        self.m_transform = nn.ModuleList(self.m_transform)
        self.m_output_act = nn.ModuleList(self.m_output_act)
        self.m_output = nn.ModuleList(self.m_output)
        self.m_frontend = nn.ModuleList(self.m_frontend)
        self.alignment = nn.ModuleList(self.alignment)

    def prepare_mean_std(self, in_dim, out_dim, args = None, data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean

    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch
        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))
            
            #  permute to (batch, frame_feat_dim, frame_num)
            x_sp_amp = x_sp_amp.permute(0, 2, 1)
            
            # make sure the buffer is long enough
            x_sp_amp_buff = torch.zeros(
                [x_sp_amp.shape[0], x_sp_amp.shape[1], trunc_len], 
                dtype=x_sp_amp.dtype, device=x_sp_amp.device)
            
            # for batch of data, pad or trim each trial independently
            fs = self.frame_hops[idx]
            for fileidx in range(x_sp_amp.shape[0]):
                # roughtly this is the number of frames
                true_frame_num = datalength[fileidx] // fs
                
                if true_frame_num > trunc_len:
                    # trim randomly
                    pos = torch.rand([1]) * (true_frame_num-trunc_len)
                    pos = torch.floor(pos[0]).long()
                    tmp = x_sp_amp[fileidx, :, pos:trunc_len+pos]
                    x_sp_amp_buff[fileidx] = tmp
                else:
                    rep = int(np.ceil(trunc_len / true_frame_num)) 
                    tmp = x_sp_amp[fileidx, :, 0:true_frame_num].repeat(1, rep)
                    x_sp_amp_buff[fileidx] = tmp[:, 0:trunc_len]

            #  permute to (batch, frame_num, frame_feat_dim)
            x_sp_amp = x_sp_amp_buff.permute(0, 2, 1)
            
        # return
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros([batch_size * self.v_submodels, 
                                  self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)
        output_domain = torch.zeros(
                [batch_size * self.v_submodels, self.alignment_dim],
                device = x.device, dtype = x.dtype
                )
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_output_act, m_output, alignment) in enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, self.v_truncate_lens,
                    self.m_transform, self.m_output_act, self.m_output, self.alignment)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)
            #  print(x_sp_amp.shape)
            #  print(x_sp_amp.unsqueeze(1).shape)
            #  set_trace()
            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))
            #  3. flatten and transform through output function
            cm_embedding = m_output_act(torch.flatten(hidden_features, 1))
            tmp_score = m_output(cm_embedding)
            tmp_domain_score = alignment(cm_embedding)
            
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_score
            output_domain[idx * batch_size : (idx + 1) * batch_size] = tmp_domain_score

        return output_emb, output_domain

    def _compute_score(self, feature_vec, inference=False):
        """
        """
        # feature_vec is [batch * submodel, 1]
        if inference:
            return feature_vec.squeeze(1)
        else:
            return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, audio, datalength, target_vec = None):
        output_score, output_domain_score = self._compute_embedding(audio, datalength)
        scores = self._compute_score(output_score)
        domain_scores = output_domain_score
        return scores, domain_scores
        #  return [scores, target_vec, True]

    def inference(self, audio, datalength, filenames, target):
        feature_vec, _ = self._compute_embedding(audio, datalength)
        scores = self._compute_score(feature_vec, True)
        
        print("Output, %s, %d, %f" % (filenames[0], target[0], scores.mean()))
        # don't write output score as a single file
        return None

if __name__ == '__main__':
    model = LCNN(1, 1, 1.0, 10, None)
    waveform = torch.randn(4, 45000, 1, dtype = torch.float32)
    print(waveform.shape)
    scores, domain_scores = model._compute_embedding(waveform, [32000,33000,34000,45000])
    print(scores.shape)
    print(domain_scores.shape)
