import torch
import numpy as np

from utils import calculate_score

class Ensembler:
    
    def __init__(self, projector, A_t, num_devices, channel_snr_db, participation_probability, client_output, task) -> None:
        self.projector = projector
        self.num_devices = num_devices
        self.participation_probability = participation_probability
        self.A_t = A_t
        self.channel_snr_db = channel_snr_db
        
        self.channel_snr = 10 ** (0.1 * self.channel_snr_db)
        self.client_output = client_output
        
        self.task = task
        
        self.Pavg = 1.0
        
    def forward(self, method, *args):
        
        return getattr(self, f"forward_{method}")(*args)
    
    def find_mu_fp(self, val_beliefs, weights):
        
        res = []
        for device_idx in range(self.num_devices):
            r = self.client_model(val_beliefs[device_idx], weights[device_idx])
            r = self.projector.project_only(r)
            r = (r ** 2).sum(dim=1).mean(dim=0)
            res.append(r)
        
        res = torch.stack(res, dim=0).mean()
        
        return res
    
    def get_gamma(self, num_participating_clients, mu_fp):
        
        mu_h = 1
        var_client = self.projector.get_sigma_client(num_participating_clients, self.num_devices) ** 2
        
        gamma = torch.sqrt(self.Pavg / (mu_h * (mu_fp + self.projector.num_dims * var_client)))
        
        return gamma
    
    def forward_oac(self, beliefs, val_beliefs, y_val_true):
        participating_devices = self.sample_participating_devices()
        weights = self.find_weights(val_beliefs, y_val_true)
        #mu_fp = self.find_mu_fp(val_beliefs, weights)
        num_participating_devices = len(participating_devices)
        
        #gamma = self.get_gamma(num_participating_devices, mu_fp)
        
        res = []
        for device_idx in participating_devices:
            r = self.client_model(beliefs[device_idx], weights[device_idx])
            
            r = self.projector.forward(r, num_participating_devices=num_participating_devices, num_devices=self.num_devices)

            r = self.A_t * r / num_participating_devices
            
            r = r / num_participating_devices
            
            res.append(r)
        
        received_signal = self.air_sum(res)
        
        y_test_pred = self.server_model(received_signal)
        
        return y_test_pred
    
    def forward_orthogonal(self, beliefs, val_beliefs, y_val_true):
        participating_devices = self.sample_participating_devices()
        weights = self.find_weights(val_beliefs, y_val_true)
        num_participating_devices = len(participating_devices)
        num_classes = beliefs[0].shape[1]
        
        res = []
        for device_idx in participating_devices:
            r = self.client_model(beliefs[device_idx], weights[device_idx])

            r = self.projector.forward(r, num_participating_devices=1, num_devices=1)

            r = self.A_t * r
            
            res.append(r)
        
        num_dims = res[0].shape[1]
        received_signal = torch.cat(res, dim=1)
        
        final_signal = torch.zeros_like(res[0])

        for i in range(num_participating_devices):
            cur_signal = received_signal[:, i*num_dims:(i+1)*num_dims]
            
            final_signal += self.add_channel_noise(cur_signal, self.channel_snr)
        
        final_signal = final_signal / num_participating_devices
        
        y_test_pred = self.server_model(final_signal)
        
        return y_test_pred
    
    def find_best_device(self, val_beliefs, y_val_true):
        
        cur_best_valscore = -np.inf 
        best_device_idx = None
        for device_idx in range(self.num_devices):
            y_val_pred = val_beliefs[device_idx].argmax(dim=1)
            valscore = calculate_score(y_val_true, y_val_pred)

            if valscore > cur_best_valscore:
                cur_best_valscore = valscore
                best_device_idx = device_idx
        
        return best_device_idx
    
    def find_weights(self, val_beliefs, y_val_true):
        
        correct_preds = torch.empty(self.num_devices, val_beliefs[0].shape[1], dtype=torch.int)
        num_data = y_val_true.shape[0]
        y_val_true = torch.nn.functional.one_hot(y_val_true, val_beliefs[0].shape[1])
        
        for device_idx in range(self.num_devices):
            y_val_pred = torch.nn.functional.one_hot(val_beliefs[device_idx].argmax(dim=1), val_beliefs[device_idx].shape[1])
            true_indices = (y_val_true == y_val_pred)
            
            correct_preds[device_idx, :] = true_indices.sum(dim=0)
        
        weights = correct_preds / num_data
        
        
        return weights
        
    def forward_bestmodel(self, beliefs, val_beliefs, y_val_true):
        
        device_idx = self.find_best_device(val_beliefs, y_val_true)
        
        r = self.client_model(beliefs[device_idx])

        r = self.projector.forward(r, num_participating_devices=1, num_devices=1)

        r = self.A_t * r

        r = r
        
        received_signal = self.add_channel_noise(r, self.channel_snr) # air_sum(client_beliefs, channel_snr)

        y_test_pred = self.server_model(received_signal)

        return y_test_pred

    def sample_participating_devices(self):

        participating_devices = []
        for device_idx in range(self.num_devices):
            rnd = np.random.uniform(0, 1)
            if rnd < self.participation_probability:
                participating_devices.append(device_idx)
        
        if len(participating_devices) == 0:
            participating_devices.append(np.random.choice(list(range(self.num_devices))))
        
        return participating_devices

    def server_model(self, signal):
        signal = signal / self.A_t
        
        signal = self.projector.invert(signal)
                
        if self.task == "multiclass":
            signal = torch.nn.functional.one_hot(signal.argmax(dim=1), signal.shape[1]) #(signal > 0.5).int()
        elif self.task == "multilabel":
            signal = (signal > 0.5).int()
        else:
            raise NotImplementedError
        
        return signal

    def client_model(self, beliefs, client_weights=None):
        num_classes = beliefs.shape[1]
        
        if self.client_output == "label":
            beliefs = torch.nn.functional.one_hot(beliefs.argmax(dim=1), num_classes)
        elif self.client_output =="belief":
            beliefs = torch.nn.functional.softmax(beliefs, dim=1)
        elif self.client_output == "weighted_belief":
            beliefs = client_weights * torch.nn.functional.softmax(beliefs, dim=1)
            beliefs = beliefs / beliefs.sum(dim=1, keepdim=True)
        else:
            raise NotImplementedError
        
        return beliefs.float()
    
    def air_sum(self, signals):

        max_sigma_channel = -1
        for signal in signals:
            sigma = self.calculate_sigma_channel(signal, self.channel_snr)
            max_sigma_channel = max(max_sigma_channel, sigma)

        signal = torch.sum(torch.stack(signals, dim=0), dim=0)
        
        signal = self.add_channel_noise_with_std(signal, max_sigma_channel)
        
        return signal
    
    def calculate_sigma_channel(self, signal, channel_snr):

        return torch.sqrt( torch.mean((signal ** 2)) / channel_snr )

    def add_channel_noise_with_std(self, signal, std):
        res = signal + torch.normal(0, std, size=signal.shape)

        return res

    def add_channel_noise(self, signal, channel_snr):
        sigma_channel = torch.sqrt( torch.mean((signal ** 2)) / channel_snr )
            
        res = signal + torch.normal(0, sigma_channel, signal.shape)

        return res
