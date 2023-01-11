import collections

import torch
import torch.nn as nn


def paras_summary(input_size, model):
    def register_hook(module):
        def hook(module_, input_, output):
            class_name = str(module_.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = collections.OrderedDict()
            summary[m_key]['input_shape'] = list(input_[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module_, 'weight'):
                params += torch.prod(torch.LongTensor(list(module_.weight.size())))
                if module_.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module_, 'bias'):
                params += torch.prod(torch.LongTensor(list(module_.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(1, *in_size) for in_size in input_size]
    else:
        x = torch.rand(1, *input_size)

    # create properties
    summary = collections.OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary
