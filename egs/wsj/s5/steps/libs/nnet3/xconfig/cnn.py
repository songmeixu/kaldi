# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.


""" This module contains the implementation of the TDNN layer.
"""

import libs.nnet3.xconfig.utils as xutils
from libs.nnet3.xconfig.basic_layers import XconfigBasicLayer
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

class XconfigConvLayer(XconfigBasicLayer):
    """This class is for parsing lines like
    conv-layer name=cnn1 filters=16 kernel=[9,9] step=[1,1] pad=false
    conv-bn-layer name=cnn1 filters=16 kernel=[9,9] stride=[1,1] pad=false
    """

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in [ 'conv-layer', 'conv-bn-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)


    def set_default_configs(self):

        super(XconfigConvLayer, self).set_default_configs()

        self.config['filters'] = 0
        self.config['kernel'] = [0, 0]
        self.config['step'] = [0, 0]
        self.config['pad'] = False

        self.config['input_dim'] = [0, 0, 0]
        self.config['output_dim'] = [0, 0, 0]
        self.config['vectorization'] = ""

    def set_configs(self, key_to_value, all_layers):
        XconfigLayerBase.set_configs(self, key_to_value, all_layers)

        # print type(all_layers[-1])
        if all_layers[-1].layer_type == "input":
            self.config['vectorization'] = "yzx"
            self.config['input_dim'][0] = self.descriptors['input']['dim'] / all_layers[-1].config['dim']
            self.config['input_dim'][1] = all_layers[-1].config['dim']
            self.config['input_dim'][2] = 1
        elif all_layers[-1].config.has_key('kernel'):  # conv or pool
            self.config['vectorization'] = "zyx"
            self.config['input_dim'][0] = all_layers[-1].config['output_dim'][0]
            self.config['input_dim'][1] = all_layers[-1].config['output_dim'][1]
            self.config['input_dim'][2] = all_layers[-1].config['output_dim'][2]

        self.config['output_dim'][0] = (1 + (self.config['input_dim'][0] - self.config['kernel'][0]) / self.config['step'][0])
        self.config['output_dim'][1] = (1 + (self.config['input_dim'][1] - self.config['kernel'][1]) / self.config['step'][1])
        self.config['output_dim'][2] = self.config['filters']
        self.config['dim'] = self.config['output_dim'][0] * \
                             self.config['output_dim'][1] * \
                             self.config['output_dim'][2]

    def set_derived_configs(self):
        """This is expected to be called after set_configs and before
        check_configs().
        """

    def check_configs(self):

        for key in ['filters']:
            if self.config[key] <= 0 or self.config['input_dim'][0] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

    def output_dim(self, auxiliary_output = None):
        return self.config['dim']

    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        # ignore the first 'tdnn' and the last 'layer'
        batchnorm = split_layer_name[1]
        pool = split_layer_name[-2]

        configs = []
        configs.append("### Begin CNN layer '{0}'".format(self.name))
        configs.append("component name={0} type=ConvolutionComponent "
                       "input-x-dim={1} input-y-dim={2} input-z-dim={3} "
                       "filt-x-dim={4} filt-y-dim={5} "
                       "filt-x-step={6} filt-y-step={7} "
                       "num-filters={8} input-vectorization-order={9} "
                       .format(self.name,
                               self.config['input_dim'][0], self.config['input_dim'][1], self.config['input_dim'][2],
                               self.config['kernel'][0], self.config['kernel'][1],
                               self.config['step'][0], self.config['step'][1],
                               self.config['filters'], self.config['vectorization']))

        # add batchnorm component
        if batchnorm == 'bn':
            configs.append("component name={0}.bn type=BatchNormComponent dim={1}"
                           .format(self.name, self.config['dim']))

        # add pool component
        if pool == 'pool':
            self.output_x = (1 + (self.input_x - self.config['kernel'][0]) / self.config['step'][0])
            self.output_y = (1 + (self.input_y - self.config['kernel'][1]) / self.config['step'][1])
            self.output_z = 1
            self.output_dim = self.output_x * self.output_y * self.output_z
            configs.append("component name={0}.bn type=BatchNormComponent dim={1}"
                           .format(self.name, self.output_dim))
            conv_vectorization = "yzx"
            configs.append("### Begin CNN layer '{0}'".format(self.name))
            configs.append("component name={0} type=MaxpoolingComponent "
                           "input-x-dim={1} input-y-dim={2} input-z-dim={3} "
                           "pool-x-size={4} pool-y-size={5} pool-z-size={6} "
                           "pool-x-step={7} pool-y-step={8} pool-z-step={9}"
                           "num-filters={8} input-vectorization-order={9} "
                           .format(self.name,
                                   self.input_x, self.input_y, self.input_z,
                                   self.config['kernel'][0], self.config['kernel'][1],
                                   self.config['step'][0], self.config['step'][1],
                                   self.config['filters'], conv_vectorization))

        # configs.append('component name=final-affine type=NaturalGradientAffineComponent '
        #       'input-dim={0} output-dim={1} param-stddev=0 bias-stddev=0'.format(
        #     nonlin_output_dim, args.num_targets)
        # configs.append('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
        #     args.num_targets)

        configs.append('# Now for the network structure')
        configs.append('component-node name={0} component={1} input={1}'
                       .format(self.name, self.name, self.descriptors['input']['final-string']))

        if batchnorm == 'bn':
            configs.append('component-node name={0}.bn component={1}.bn input={2}'
                           .format(self.name, self.name, self.name))
            # configs.append('component-node name=final-affine component=final-affine input={0}.bn'
            #                .format(self.name))
        # else:
            # configs.append('component-node name=final-affine component=final-affine input={0}'
            #                .format(self.name))

        # configs.append('component-node name=final-log-softmax component=final-log-softmax '
        #                'input=final-affine')
        # configs.append('output-node name=output input=final-log-softmax')

        return configs



class XconfigPoolLayer(XconfigBasicLayer):
    """This class is for parsing lines like
    max-pool-layer name=pool1 kernel=[2,2] step=[2,2] pad=False
    """

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in [ 'max-pool-layer', 'avg-pool-layer' ]
        # TODO: 'avg-pool-layer' to be implemented in c++
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        super(XconfigPoolLayer, self).set_default_configs()

        self.config['kernel'] = [0, 0]
        self.config['step'] = [0, 0]
        self.config['pad'] = False

        self.config['input_dim'] = [0, 0, 0]
        self.config['output_dim'] = [0, 0, 0]

    def set_configs(self, key_to_value, all_layers):
        XconfigLayerBase.set_configs(self, key_to_value, all_layers)

        if all_layers[-1].config.has_key('kernel'): # prev is conv
            prev_conv = all_layers[-1]
        else:
            raise RuntimeError("{0} previous layer is not convolution layer."
                               .format(self.name))

        self.config['input_dim'][0] = prev_conv.config['output_dim'][0]
        self.config['input_dim'][1] = prev_conv.config['output_dim'][1]
        self.config['input_dim'][2] = prev_conv.config['output_dim'][2]

        self.config['output_dim'][0] = (1 + (self.config['input_dim'][0] - self.config['kernel'][0]) / self.config['step'][0])
        self.config['output_dim'][1] = (1 + (self.config['input_dim'][1] - self.config['kernel'][1]) / self.config['step'][1])
        self.config['output_dim'][2] = self.config['input_dim'][2]
        self.config['dim'] = self.config['output_dim'][0] * \
                             self.config['output_dim'][1] * \
                             self.config['output_dim'][2]

    def set_derived_configs(self):
        """This is expected to be called after set_configs and before
        check_configs().
        """

    def check_configs(self):

        for key in ['kernel']:
            if self.config[key][0] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))


    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        self.pool_type = split_layer_name[0]

        configs = []
        # add pool component
        configs.append("### Begin Pooling layer '{0}'".format(self.name))
        configs.append("component name={0} type=MaxpoolingComponent "
                       "input-x-dim={1} input-y-dim={2} input-z-dim={3} "
                       "pool-x-size={4} pool-y-size={5} pool-z-size={6} "
                       "pool-x-step={7} pool-y-step={8} pool-z-step={9}"
                       .format(self.name,
                               self.config['input_dim'][0], self.config['input_dim'][1], self.config['input_dim'][2],
                               self.config['kernel'][0], self.config['kernel'][1], 1,
                               self.config['step'][0], self.config['step'][1], 1))

        configs.append('# Now for the network structure')
        configs.append('component-node name={0} component={1} input={2}'
                       .format(self.name, self.name, self.descriptors['input']['final-string']))

        return configs
