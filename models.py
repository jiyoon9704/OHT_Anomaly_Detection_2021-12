import torch
import torch.nn as nn

"""Fully-Connected Autoencoder(FCAE)"""

class FCAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, mid_act = nn.ReLU(), final_layer = True): #1200, 256, 128
        super(FCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),    
            mid_act,
            nn.Linear(hidden_size1, hidden_size2),
            mid_act
        )
        if final_layer:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,                        
                nn.Linear(hidden_size1, input_size),
                nn.Sigmoid()                          
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,                 
                nn.Linear(hidden_size1, input_size)
                #nn.Sigmoid()                          
            )

                
    def forward(self, x, c=0):              
        z = self.encoder(x)           
        x_hat = self.decoder(z)            
        return x_hat, z
    
"""Conditoinal Autoencoder(CondAE)"""

class CondAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, c_dim, mid_act = nn.ReLU(), final_layer = True): #1200, 256, 128
        super(CondAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size+c_dim, hidden_size1),    
            mid_act,
            nn.Linear(hidden_size1, hidden_size2),
            mid_act
        )
        if final_layer:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size2+c_dim, hidden_size1),
                mid_act,                        
                nn.Linear(hidden_size1, input_size),
                nn.Sigmoid()                          
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size2+c_dim, hidden_size1),
                mid_act,                 
                nn.Linear(hidden_size1, input_size)
                #nn.Sigmoid()                          
            )

                
    def forward(self, x, c):  
        concat_input = torch.cat([x, c], 1)
        z = self.encoder(concat_input)
        x_hat = self.decoder(torch.cat([z, c], 1))            
        return x_hat, z
    
    
"""RNN-based Autoencoder(RAE)"""
    
class RNNEncoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, block, dropout=0): #256,256
        super(RNNEncoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=self.n_features, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=self.n_features, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError

    def forward(self,x):

        #x.size() = (batch_size*time_window *n_features)
        batch_size = x.size()[0]

        if isinstance(self.model, nn.LSTM):
            _, (h_end, c_end) = self.model(x)
            
        elif isinstance(self.model, nn.GRU) or isinstance(self.model, nn.RNN):
            _, h_end = self.model(x)
        h_end = h_end[-1,:, :]
        
        return h_end.reshape((batch_size,  self.hidden_dim))  
    
class RNNDecoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, block, dropout=0, final_layer=False): #256, 256
        super(RNNDecoder, self).__init__()

        self.seq_len = seq_len  
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.final_layer = final_layer
        if self.final_layer:
            self.sigmoid = nn.Sigmoid()

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError
            
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)
        

    def forward(self, x):
        batch_size = x.size(0) # x shape(256,256)
        
        decoder_input = torch.stack([x for _ in range(self.seq_len)], dim = 1) # (256, 256) -> (256,100,256)
        
        decoder_output, _ = self.model(decoder_input)
        
        out = self.output_layer(decoder_output)

        if self.final_layer:
            out = nn.Sigmoid(out)
        
        return out
    
class RAE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size, RNN_type ="LSTM", final_layer = True): #12, 100, 256, "LSTM"
        super(RAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = RNNEncoder(self.seq_len, self.n_features, self.hidden_size, block=RNN_type)
        self.decoder = RNNDecoder(self.seq_len, self.n_features, self.hidden_size, block=RNN_type, final_layer = final_layer)
        
    def forward(self, x, c=0):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    

"""Conditional RNN-based Autoencoder(CRAE)"""

class CRAEEncoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, c_dim, block='LSTM', dropout=0): #256,256
        super(CRAEEncoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError

    def forward(self, x, c):

        #x.size() = (batch_size*time_window *n_features)
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        c_repeat = torch.stack([c for _ in range(seq_len)], dim = 1)

        concat_input = torch.cat((x, c_repeat), dim=2)
        if isinstance(self.model, nn.LSTM):
            _, (h_end, c_end) = self.model(concat_input)
            
        elif isinstance(self.model, nn.GRU) or isinstance(self.model, nn.RNN):
            _, h_end = self.model(concat_input)
        h_end = h_end[-1,:, :]
        return h_end.reshape((batch_size,  self.hidden_dim))  

    
class CRAEDecoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, c_dim, block='LSTM', dropout=0, final_layer = True): #256,256
        super(CRAEDecoder, self).__init__()

        self.seq_len = seq_len  
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.final_layer = final_layer
        if self.final_layer:
            self.sigmoid = nn.Sigmoid()

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=self.hidden_dim+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=self.hidden_dim+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=self.hidden_dim+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError
            
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

    def forward(self, x, c):
        batch_size = x.size(0) # x shape(256, 128)
        
        decoder_inputs = torch.stack([x for _ in range(self.seq_len)], dim = 1) # (256, 256) -> (256,100,256)

        c_repeat = torch.stack([c for _ in range(self.seq_len)], dim = 1) #(256,9) -> (256,100,9)
        concat_input = torch.cat((decoder_inputs, c_repeat), dim=2)
        
        decoder_output, _ = self.model(concat_input)
        
        out = self.output_layer(decoder_output)
        
        if self.final_layer:
            out = self.sigmoid(out)
        
        return out
    
    
class CRAE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size, n_labels=9, RNN_type ="LSTM", final_layer = True):
        super(CRAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = CRAEEncoder(self.seq_len, self.n_features, self.hidden_size, n_labels, block=RNN_type)
        self.decoder = CRAEDecoder(self.seq_len, self.n_features, self.hidden_size, n_labels, block=RNN_type, final_layer = final_layer)
    def forward(self, x, c):
        z = self.encoder(x,c)
        x_hat = self.decoder(z, c)
        return x_hat, z
    
    
"""Convolutional Autoencoder(CAE)"""

class CAE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size1, hidden_size2, mid_act=nn.ReLU(), final_layer = True): #12, 256, 128
        super(CAE ,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_features, hidden_size2, kernel_size = [7,1], stride = 3),
            mid_act,
            nn.Conv2d(hidden_size2, hidden_size1, kernel_size=[3,1], stride = 2),
            mid_act,
            nn.Conv2d(hidden_size1, hidden_size1, kernel_size=[3,1], stride = 2),
            mid_act,
            nn.Conv2d(hidden_size1, hidden_size2, kernel_size=[3,1], stride = 2),
            mid_act
        )
        if final_layer:
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_size2, hidden_size2, kernel_size=[3,1], stride = 1, padding=(1,0)),
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size2, hidden_size1, kernel_size=[3,1], stride = 1, padding=(1,0)), 
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size1, hidden_size2, kernel_size=[3,1], stride = 1, padding=(1,0)),  
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size2, n_features, kernel_size=[3,1], stride = 1, padding=(1,0)),
                nn.Upsample(size= (seq_len, 1), mode='nearest'),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_size2, hidden_size2, kernel_size=[3,1], stride = 1, padding=(1,0)),
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size2, hidden_size1, kernel_size=[3,1], stride = 1, padding=(1,0)), 
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size1, hidden_size2, kernel_size=[3,1], stride = 1, padding=(1,0)),  
                mid_act,
                nn.Upsample(scale_factor=(3,1), mode='nearest'),
                nn.Conv2d(hidden_size2, n_features, kernel_size=[3,1], stride = 1, padding=(1,0)),
                nn.Upsample(size= (seq_len, 1), mode='nearest')
            )

                
    def forward(self, x, c = 0):               
        z = self.encoder(x)          
        x_hat = self.decoder(z)      
        return x_hat, z
    
    
"""Variational Autoencoder(VAE)"""

class VAE(nn.Module):  
    def __init__(self, input_size, hidden_size1, hidden_size2, mid_act = nn.ReLU(), final_layer = True, latent_size = 10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),    
            mid_act,                        
            nn.Linear(hidden_size1, hidden_size2),
            mid_act,                           
        )
        
        self.fc_mu = nn.Linear(hidden_size2, latent_size)
        self.fc_var = nn.Linear(hidden_size2, latent_size)
        
        if final_layer:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, hidden_size2),
                mid_act,
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,   
                nn.Linear(hidden_size1, input_size),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size, hidden_size2),
                mid_act,
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,   
                nn.Linear(hidden_size1, input_size)
            )
                
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        recon = self.decoder(z)
        return recon
    
                
    def forward(self, x, c = 0):             
        batch_size = x.size(0)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z
    
    
"""Conditional Variational Autoencoder(CVAE)"""

class CVAE(nn.Module):  
    def __init__(self, input_size, hidden_size1, hidden_size2, cond_size = 9, mid_act = nn.ReLU(), final_layer = True, latent_size = 10):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size+cond_size, hidden_size1),    
            mid_act,
            nn.Linear(hidden_size1, hidden_size2),
            mid_act
        )
        
        self.fc_mu = nn.Linear(hidden_size2, latent_size)
        self.fc_var = nn.Linear(hidden_size2, latent_size)
        
        if final_layer:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size+cond_size, hidden_size2),
                mid_act,
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,
                nn.Linear(hidden_size1, input_size),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_size+cond_size, hidden_size2),
                mid_act,
                nn.Linear(hidden_size2, hidden_size1),
                mid_act,
                nn.Linear(hidden_size1, input_size)
            )
                
    def encode(self, x, c):
        concat_input = torch.cat([x, c], 1)
        h = self.encoder(concat_input)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, c):
        concat_input = torch.cat([z, c], 1)
        recon = self.decoder(concat_input)
        return recon
    
                
    def forward(self, x, c):             
        batch_size = x.size(0)
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, c)
        return x_hat, mu, log_var, z


"""Augmented RNN-based Autoencoder(CRAE)"""

class ARAEEncoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, c_dim, block='LSTM', dropout=0): #256,256
        super(ARAEEncoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=n_features+c_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError

    def forward(self, x, c):

        #x.size() = (batch_size*time_window *n_features)
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        c_repeat = torch.stack([c for _ in range(seq_len)], dim = 1)

        concat_input = torch.cat((x, c_repeat), dim=2)
        if isinstance(self.model, nn.LSTM):
            _, (h_end, c_end) = self.model(concat_input)
            
        elif isinstance(self.model, nn.GRU) or isinstance(self.model, nn.RNN):
            _, h_end = self.model(concat_input)
        h_end = h_end[-1,:, :]
        return h_end.reshape((batch_size,  self.hidden_dim))  

    
class ARAEDecoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim, c_dim, block='LSTM', dropout=0, final_layer = True): #256,256
        super(ARAEDecoder, self).__init__()

        self.seq_len = seq_len  
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.final_layer = final_layer
        if self.final_layer:
            self.sigmoid = nn.Sigmoid()

        if block == 'LSTM':
            self.model = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        elif block == "RNN":
            self.model = nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True, nonlinearity = "relu")
        elif block == 'GRU':
            self.model = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim,num_layers=2,dropout = dropout, 
                                 batch_first = True)
        else:
            raise NotImplementedError
            
        self.output_layer = nn.Linear(self.hidden_dim, self.n_features+c_dim)

    def forward(self, x, c):
        batch_size = x.size(0) # x shape(256, 128)
        
        decoder_inputs = torch.stack([x for _ in range(self.seq_len)], dim = 1) # (256, 256) -> (256,100,256)
        
        decoder_output, _ = self.model(decoder_inputs)
        
        out = self.output_layer(decoder_output)
        
        if self.final_layer:
            out = self.sigmoid(out)
        
        return out
    
    
class ARAE(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size, n_labels=9, RNN_type ="LSTM", final_layer = True):
        super(ARAE, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = ARAEEncoder(self.seq_len, self.n_features, self.hidden_size, n_labels, block=RNN_type)
        self.decoder = ARAEDecoder(self.seq_len, self.n_features, self.hidden_size, n_labels, block=RNN_type, final_layer = final_layer)
    def forward(self, x, c):
        z = self.encoder(x,c)
        xc_hat = self.decoder(z, c)
        return xc_hat, z

