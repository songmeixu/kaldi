# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.

""" This module contains the parent class from which all layers are inherited
and some basic layer definitions.
"""

import libs.nnet3.xconfig.utils as xutils
from libs.nnet3.xconfig.basic_layers import XconfigBasicLayer
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is for parsing lines like
#  'relu-renorm-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# or:
#  'sigmoid-layer name=layer1 dim=1024 input=Append(-3,0,3)'
# which specify addition of an affine component and a sequence of non-linearities.
# Here, the name of the layer itself dictates the sequence of nonlinearities
# that are applied after the affine component; the name should contain some
# combination of 'relu', 'renorm', 'sigmoid' and 'tanh',
# and these nonlinearities will be added along with the affine component.
#
# The dimension specified is the output dim; the input dim is worked out from the input descriptor.
# This class supports only nonlinearity types that do not change the dimension; we can create
# another layer type to enable the use p-norm and similar dimension-reducing nonlinearities.
#
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=None                   [Output dimension of layer, e.g. 1024]
#   self-repair-scale=1.0e-05  [Affects relu, sigmoid and tanh layers.]
#
class XconfigBNNBasicLayer(XconfigBasicLayer):
    def __init__(self, first_token, key_to_value, prev_names = None):
        # Here we just list some likely combinations.. you can just add any
        # combinations you want to use, to this list.
        # bw: binary_weight bn: batch-norm ba: binary_activation
        assert first_token in [ 'bnn-bn-ba-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        nonlinearities = split_layer_name[1:-2]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']

        # the child classes e.g. tdnn might want to process the input
        # before adding the other components

        return self._add_components(input_desc, input_dim, nonlinearities)

    def _add_components(self, input_desc, input_dim, nonlinearities):
        output_dim = self.output_dim()
        self_repair_scale = self.config['self-repair-scale']
        target_rms = self.config['target-rms']
        max_change = self.config['max-change']
        ng_opt_str = self.config['ng-affine-options']

        configs = []
        # First the affine node.
        line = ('component name={0}.affine'
                ' type=BinaryAffineComponent'
                ' input-dim={1}'
                ' output-dim={2}'
                ' max-change={3}'
                ' {4}'
                ''.format(self.name, input_dim, output_dim,
                    max_change, ng_opt_str))
        configs.append(line)

        line = ('component-node name={0}.affine'
                ' component={0}.affine input={1}'
                ''.format(self.name, input_desc))
        configs.append(line)
        cur_node = '{0}.affine'.format(self.name)

        for nonlinearity in nonlinearities:
            if nonlinearity == 'relu':
                line = ('component name={0}.{1}'
                        ' type=RectifiedLinearComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'sigmoid':
                line = ('component name={0}.{1}'
                        ' type=SigmoidComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'tanh':
                line = ('component name={0}.{1}'
                        ' type=TanhComponent dim={2}'
                        ' self-repair-scale={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            self_repair_scale))

            elif nonlinearity == 'renorm':
                line = ('component name={0}.{1}'
                        ' type=NormalizeComponent dim={2}'
                        ' target-rms={3}'
                        ''.format(self.name, nonlinearity, output_dim,
                            target_rms))

            elif nonlinearity == 'bn':
                line = ('component name={0}.{1}'
                        ' type=BatchNormComponent dim={2}'
                        ''.format(self.name, nonlinearity, output_dim))

            else:
                raise RuntimeError("Unknown nonlinearity type: {0}"
                                   .format(nonlinearity))

            configs.append(line)
            line = ('component-node name={0}.{1}'
                    ' component={0}.{1} input={2}'
                    ''.format(self.name, nonlinearity, cur_node))

            configs.append(line)
            cur_node = '{0}.{1}'.format(self.name, nonlinearity)

        # last add the binary activation node.
        line = ('component name={0}.ba'
                ' type=BinaryActivitionComponent'
                ' dim={1}'
                .format(self.name, output_dim))
        configs.append(line)

        line = ('component-node name={0}.ba'
                ' component={0}.ba input={0}.{1}'
                ''.format(self.name, nonlinearities[-1]))
        configs.append(line)

        return configs

def test_layers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(config_line_to_object(x, [])) == x
