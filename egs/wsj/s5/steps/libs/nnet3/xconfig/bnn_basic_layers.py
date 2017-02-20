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
        norm_history = self.config['norm-history']

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
                        ' type=BatchNormComponent dim={2} norm-history={3}'
                        ''.format(self.name, nonlinearity, output_dim, norm_history))

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

class XconfigBNNOutputLayer(XconfigLayerBase):
    """This class is for lines like
    'bnn-output-layer name=output dim=4257 input=Append(input@-1, input@0, input@1, ReplaceIndex(ivector, t, 0))'
    By default this includes a log-softmax component.  The parameters are
    initialized to zero, as this is best for output layers.

    Parameters of the class, and their defaults:
        input='[-1]'    :   Descriptor giving the input of the layer.
        dim=None    :   Output dimension of layer, will normally equal the number of pdfs.
        include-log-softmax=true    :   setting it to false will omit the
            log-softmax component- useful for chain models.
        objective-type=linear   :   the only other choice currently is
            'quadratic', for use in regression problems
        learning-rate-factor=1.0    :   Learning rate factor for the final
            affine component, multiplies the standard learning rate. normally
            you'll leave this as-is, but for xent regularization output layers
            for chain models you'll want to set
            learning-rate-factor=(0.5/xent_regularize),
            normally learning-rate-factor=5.0 since xent_regularize is
            normally 0.1.
        presoftmax-scale-file=None  :   If set, a filename for a vector that
            will be used to scale the output of the affine component before the
            log-softmax (if include-log-softmax=true), or before the output
            (if not).  This is helpful to avoid instability in training due to
            some classes having much more data than others.  The way we normally
            create this vector is to take the priors of the classes to the
            power -0.25 and rescale them so the average is 1.0.  This factor
            -0.25 is referred to as presoftmax_prior_scale_power in scripts. In
            the scripts this would normally be set to
            config_dir/presoftmax_prior_scale.vec
    """

    def __init__(self, first_token, key_to_value, prev_names = None):

        assert first_token in [ 'bnn-output-layer', 'bnn-bn-output-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):

        # note: self.config['input'] is a descriptor, '[-1]' means output
        # the most recent layer.
        self.config = {'input' : '[-1]',
                       'dim' : -1,
                       'include-log-softmax' : True,
                       # this would be false for chain models
                       'objective-type' : 'linear',
                       # see Nnet::ProcessOutputNodeConfigLine in
                       # nnet-nnet.cc for other options
                       'learning-rate-factor' : 1.0,
                       'presoftmax-scale-file' : '',
                       # used in DNN (not RNN) training when using
                       # frame-level objfns,
                       'max-change' : 1.5,
                       'param-stddev' : 0.0,
                       'bias-stddev' : 0.0,
                       'output-delay' : 0,
                       }

    def set_derived_configs(self):
        super(XconfigBNNOutputLayer, self).set_derived_configs()
        if self.config['param-stddev'] < 0:
            self.config['param-stddev'] = 1.0 / math.sqrt(self.descriptors['input']['dim'])

    def check_configs(self):

        if self.config['dim'] <= -1:
            raise RuntimeError("In bnn-output-layer, dim has invalid value {0}"
                               "".format(self.config['dim']))

        if self.config['objective-type'] != 'linear' and \
                        self.config['objective_type'] != 'quadratic':
            raise RuntimeError("In bnn-output-layer, objective-type has"
                               " invalid value {0}"
                               "".format(self.config['objective-type']))

        if self.config['learning-rate-factor'] <= 0.0:
            raise RuntimeError("In bnn-output-layer, learning-rate-factor has"
                               " invalid value {0}"
                               "".format(self.config['learning-rate-factor']))


    # you cannot access the output of this layer from other layers... see
    # comment in output_name for the reason why.
    def auxiliary_outputs(self):

        return []

    def output_name(self, auxiliary_outputs = None):

        # Note: nodes of type output-node in nnet3 may not be accessed in
        # Descriptors, so calling this with auxiliary_outputs=None doesn't
        # make sense.  But it might make sense to make the output of the softmax
        # layer and/or the output of the affine layer available as inputs to
        # other layers, in some circumstances.
        # we'll implement that when it's needed.
        raise RuntimeError("Outputs of bnn-output-layer may not be used by other"
                           " layers")

    def output_dim(self, auxiliary_output = None):

        # see comment in output_name().
        raise RuntimeError("Outputs of bnn-output-layer may not be used by other"
                           " layers")

    def get_full_config(self):

        ans = []

        # note: each value of self.descriptors is (descriptor, dim,
        # normalized-string, output-string).
        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        descriptor_final_string = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        output_dim = self.config['dim']
        objective_type = self.config['objective-type']
        learning_rate_factor = self.config['learning-rate-factor']
        include_log_softmax = self.config['include-log-softmax']
        presoftmax_scale_file = self.config['presoftmax-scale-file']
        param_stddev = self.config['param-stddev']
        bias_stddev = self.config['bias-stddev']
        output_delay = self.config['output-delay']
        max_change = self.config['max-change']

        # note: ref.config is used only for getting the left-context and
        # right-context of the network;
        # final.config is where we put the actual network definition.
        for config_name in ['ref', 'final' ]:
            # First the affine node.
            line = ('component name={0}.affine'
                    ' type=BinaryAffineComponent'
                    ' input-dim={1}'
                    ' output-dim={2}'
                    ' param-stddev={3}'
                    ' bias-stddev={4}'
                    ' max-change={5} '
                    ''.format(self.name, input_dim, output_dim,
                              param_stddev, bias_stddev, max_change) +
                    ('learning-rate-factor={0} '.format(learning_rate_factor)
                     if learning_rate_factor != 1.0 else ''))
            ans.append((config_name, line))

            line = ('component-node name={0}.affine'
                    ' component={0}.affine input={1}'
                    ''.format(self.name, descriptor_final_string))
            ans.append((config_name, line))

            cur_node = '{0}.affine'.format(self.name)

            # add the batchnorm node.
            split_layer_name = self.layer_type.split('-')
            assert split_layer_name[-1] == 'layer'
            if split_layer_name[1] == 'bn':
                line = ('component name={0}.bn'
                        ' type=BatchNormComponent dim={1}'
                        ''.format(self.name, output_dim))
                ans.append((config_name, line))
                line = ('component-node name={0}.bn'
                        ' component={0}.bn input={1}'
                        ''.format(self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.bn'.format(self.name)

            if presoftmax_scale_file is not '' and config_name == 'final':
                # don't use the presoftmax-scale in 'ref.config' since that
                # file won't exist at the time we evaluate it.
                # (ref.config is used to find the left/right context).
                line = ('component name={0}.fixed-scale'
                        ' type=FixedScaleComponent scales={1}'
                        ''.format(self.name, presoftmax_scale_file))
                ans.append((config_name, line))

                line = ('component-node name={0}.fixed-scale'
                        ' component={0}.fixed-scale input={1}'
                        ''.format(self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.fixed-scale'.format(self.name)

            if include_log_softmax:
                line = ('component name={0}.log-softmax'
                        ' type=LogSoftmaxComponent dim={1}'
                        ''.format(self.name, output_dim))
                ans.append((config_name, line))

                line = ('component-node name={0}.log-softmax'
                        ' component={0}.log-softmax input={1}'
                        ''.format(self.name, cur_node))
                ans.append((config_name, line))
                cur_node = '{0}.log-softmax'.format(self.name)

            if output_delay != 0:
                cur_node = 'Offset({0}, {1})'.format(cur_node, output_delay)

            line = ('output-node name={0} input={1}'.format(self.name, cur_node))
            ans.append((config_name, line))

        return ans

def test_layers():
    # for some config lines that should be printed the same way as they
    # are read, check that this is the case.
    for x in [ 'input name=input dim=30' ]:
        assert str(config_line_to_object(x, [])) == x
