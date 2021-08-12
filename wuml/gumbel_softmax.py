import torch
from torch.distributions.uniform import Uniform

def gumbel(gumbel_input, tau=0.1, device='cpu'):
	#p = F.normalize(gumbel_input, dim=2, p=1)		# normalize to probability distribution

	uniform_dist = Uniform(1e-30, 1.0, )	
	uniform = uniform_dist.rsample(sample_shape=gumbel_input.size())
	uniform = uniform.to(device, non_blocking=True )

	ε = -torch.log(-torch.log(uniform))
	noisy_logits = (ε + gumbel_input) / tau
	samples = torch.nn.Softmax(dim=-1)(noisy_logits)
	return samples
	#return samples.transpose(-1, -2)


#def sample_from_gumbel_softmax(group_weights, k=1, tau=0.1):
#    logits_ = group_weights.view(group_weights.size(0), 1, group_weights.size(-2)*group_weights.size(-1))
#    batch_size = group_weights.size(0)
#    num_features = group_weights.size(1)
#    num_groups = group_weights.size(2)
#    samples_list = []
#    for i in range(num_features):
#        sub_logits = logits_[:,:,i*num_groups:(i+1)*num_groups]
#        uniform_dist = Uniform(1e-10, 1.0, )
#        uniform = uniform_dist.rsample(sample_shape=(batch_size, k, num_groups) )
#        gumbel = -torch.log(-torch.log(uniform))
#        noisy_logits = (gumbel + sub_logits) / tau
#        samples = torch.nn.Softmax(dim=-1)(noisy_logits)
#        # print (samples)
#        # samples = torch.max(noisy_logits, 1)
#        # print (samples)
#        samples_list.append(samples)
#    g = torch.cat(samples_list, 1)
#    return g.transpose(-1, -2)
#
#def sample_from_gumbel_softmax_parallel(gumbel_input, k=1, tau=0.1, device='cpu'):
#	batch_size = gumbel_input.size(0)
#	d = gumbel_input.size(1)
#	num_groups = gumbel_input.size(2)
#	logits_ = gumbel_input
#
#	uniform_dist = Uniform(1e-30, 1.0, )	
#	uniform = uniform_dist.rsample(sample_shape=(batch_size, k * d, num_groups))
#	uniform = uniform.to(device, non_blocking=True )
#	gumbel = -torch.log(-torch.log(uniform))
#
#	noisy_logits = (gumbel + logits_) / tau
#	samples = torch.nn.Softmax(dim=-1)(noisy_logits)
#	
#	return samples.transpose(-1, -2)
#
#
#def sample_from_gumbel_softmax_original(group_weights, k=1, tau=0.1):
#    logits_ = group_weights.unsqueeze(1)
#    batch_size = group_weights.size(0)
#    d = logits_.size(2)
#    uniform_dist = Uniform(1e-10, 1.0, )
#    uniform = uniform_dist.rsample(sample_shape=(batch_size, k, d))
#
#    gumbel = -torch.log(-torch.log(uniform))
#    noisy_logits = (gumbel + logits_) / tau
#    samples = torch.nn.Softmax(dim=-1)(noisy_logits)
#
#    return samples.max(dim=1)[0]
