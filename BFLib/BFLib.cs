using System.Buffers;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

namespace BFLib
{
    namespace AI
    {
        public class DenseNeuralNetwork
        {
            public readonly int inDim, outDim;
            public readonly Layer[] layers;
            public readonly DenseWeightMatrix[] weights;

            public DenseNeuralNetwork(params long[] dims)
            {
                using (DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder())
                {
                    builder.NewLayers(dims);

                    Tuple<Layer[], DenseWeightMatrix[]> bundle = builder.Build();

                    this.layers = bundle.Item1;
                    this.weights = bundle.Item2;
                    this.inDim = layers[0].dim;
                    this.outDim = layers[layers.Length - 1].dim;
                }
            }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, bool disposeAfterwards = true)
            {
                Tuple<Layer[], DenseWeightMatrix[]> bundle = builder.Build();

                this.layers = bundle.Item1;
                this.weights = bundle.Item2;
                this.inDim = layers[0].dim;
                this.outDim = layers[layers.Length - 1].dim;

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public void WeightAssignForEach(Func<double> func)
            {
                for (int i = 0; i < weights.Length; i++)
                    for (int j = 0; j < weights[i].matrix.GetLength(0); j++)
                        for (int k = 0; k < weights[i].matrix.GetLength(1); k++)
                            weights[i].matrix[j, k] = func();
            }

            public void BiasAssignForEach(Func<double> func)
            {
                for (int i = 0; i < layers.Length; i++)
                    for (int j = 0; j < layers[i].biases.Length; j++)
                        layers[i].biases[j] = func();
            }

            /// <summary>
            /// Backpropagates and updates weights, biases
            /// </summary>
            public void GradientDescent(double[] sampleOutputs, DenseNNForwardResult forwardLog, double learningRate)
            {
                double[] errors = new double[sampleOutputs.Length];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int i = 0; i < sampleOutputs.Length; i++)
                    errors[i] = forwardLog.outputs[i] - sampleOutputs[i];

                GradientDescentLayers(errors, forwardLog, learningRate, layers.Length - 1);
            }

            /// <summary>
            /// <i>For recursion purpose only</i>. Backpropagates and updates specified layers
            /// </summary>
            /// <param name="errors">Error vector of the commencing layer (or <b>fromLayer</b>)</param>
            /// <param name="fromlayer"><i>For recursion purpose only</i>. Going backwards from the given <b>fromLayer</b> index</param>
            private void GradientDescentLayers(double[] errors, DenseNNForwardResult forwardLog, double learningRate, int fromLayer)
            {
                if (fromLayer < 1)
                    return;

                double[] layerErrors = new double[layers[fromLayer - 1].dim];

                for (int i = 0; i < layers[fromLayer].dim; i++)
                {
                    for (int j = 0; j < layers[fromLayer - 1].dim; j++)
                    {
                        // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                        layerErrors[j] += errors[i] * weights[fromLayer - 1].matrix[i, j] * layers[fromLayer - 1].FunctionDifferential(layers[fromLayer - 1].biases[j], forwardLog.layerInputs[fromLayer - 1][j]);

                        // weight update
                        weights[fromLayer - 1].matrix[i, j] -= learningRate * errors[i] * layers[fromLayer - 1].ForwardComp(forwardLog.layerInputs[fromLayer - 1][j], layers[fromLayer - 1].biases[j]);
                    }
                }

                for (int i = 0; i < layers[fromLayer - 1].dim; i++)
                {
                    // bias update
                    layers[fromLayer - 1].biases[i] -= learningRate * layerErrors[i];
                }

                GradientDescentLayers(layerErrors, forwardLog, learningRate, fromLayer - 1);
            }

            public DenseNNForwardResult Forward(double[] inputs)
            {
                double[][] layerInputs = new double[layers.Length][];

                ForwardLayers(inputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs);
            }

            public double[] ForwardLayers(double[] inputs, int toLayer, int fromLayer, ref double[][] layerInputs)
            {
                // Output layer (last layer) doesn't have bias
                if (toLayer == layers.Length - 1)
                {
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs)).ToArray();
                    return layerInputs[toLayer];
                }

                if (fromLayer < toLayer)
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs)).ToArray();
                else
                    layerInputs[toLayer] = inputs;

                return layers[toLayer].Forward(layerInputs[toLayer]).ToArray();
            }
        }

        public class DenseNNForwardResult
        {
            public double[][] layerInputs;

            public double[] outputs => layerInputs[layerInputs.Length - 1];

            public DenseNNForwardResult(double[][] layerInputs) 
            {
                this.layerInputs = layerInputs;
            }
        }

        public class DenseNeuralNetworkBuilder : INeuralNetworkBuilder
        {
            public List<Layer> layers;

            public DenseNeuralNetworkBuilder() 
            {
                layers = new List<Layer>();
            }

            public void NewLayers(long[] dims)
            {
                foreach (int dim in dims)
                    NewLayer(dim);
            }

            public void NewLayer(int dim)
            {
                layers.Add(new Layer(dim));
            }

            public void NewLayer(Layer layer)
            {
                Layer clone = new Layer(layer.dim);

                for (int i = 0; i < layer.dim; i++)
                    clone.biases[i] = layer.biases[i];

                layers.Add(clone);
            }

            public Tuple<Layer[], DenseWeightMatrix[]> Build()
            {
                DenseWeightMatrix[]  weights = new DenseWeightMatrix[layers.Count - 1];

                for (int i = 1; i < layers.Count; i++)
                    weights[i - 1] = new DenseWeightMatrix(layers[i - 1].dim, layers[i].dim);

                return (layers.ToArray(), weights).ToTuple();
            }

            public void Dispose()
            {
                GC.SuppressFinalize(this);
            }
        }

        public interface INeuralNetworkBuilder : IDisposable
        {
            public abstract Tuple<Layer[], DenseWeightMatrix[]> Build();
        }

        public class Layer
        {
            public readonly int dim;
            public double[] biases;

            public Layer(int dim)
            {
                this.dim = dim;
                this.biases = new double[dim];
            }
            
            public Layer(double[] biases)
            {
                this.dim = biases.Length;
                this.biases = biases;
            }

            public virtual IEnumerable<double> Forward(double[] inputs)
            {
                for (int i = 0; i < dim; i++)
                    yield return ForwardComp(inputs[i], biases[i]);
            }

            public virtual double ForwardComp(double input, double bias) => bias + input;

            /// <summary>
            /// Get <b>df(bias, x) / dx</b> such <b>x</b> can be another function
            /// </summary>
            public virtual double FunctionDifferential(double bias, double x) => 1;
        }

        public class DenseWeightMatrix
        {
            public readonly int inDim, outDim;

            public double[,] matrix;

            public DenseWeightMatrix(int inDim, int outDim)
            {
                this.inDim = inDim;
                this.outDim = outDim;
                this.matrix = new double[outDim, inDim];
            }

            public DenseWeightMatrix(double[,] matrix)
            {
                this.inDim = matrix.GetLength(1);
                this.outDim = matrix.GetLength(0);

                this.matrix = matrix;             // need testing for shallow cloned or not
            }

            public IEnumerable<double> Forward(double[] inputs)
            {
                for (int i = 0; i < outDim; i++) 
                    yield return ForwardComp(inputs, i);
            }

            public double ForwardComp(double[] inputs, int outputIndex)
            {
                double output = 0;

                for (int i = 0; i < inputs.Length; i++)
                    output += matrix[outputIndex, i] * inputs[i];

                return output;
            }
        }
    }
}