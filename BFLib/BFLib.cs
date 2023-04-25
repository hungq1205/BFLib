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

            public DenseNeuralNetwork(params Layer[] dims)
            {
                using (DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder())
                {
                    builder.NewLayers(dims);

                    Tuple<Layer[], DenseWeightMatrix[]> bundle = builder.Build();

                    this.layers = bundle.Item1;
                    this.weights = bundle.Item2;
                    this.inDim = layers[0].dim;
                    this.outDim = layers[layers.LongLength - 1].dim;
                }
            }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, bool disposeAfterwards = true)
            {
                Tuple<Layer[], DenseWeightMatrix[]> bundle = builder.Build();

                this.layers = bundle.Item1;
                this.weights = bundle.Item2;
                this.inDim = layers[0].dim;
                this.outDim = layers[layers.LongLength - 1].dim;

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public void WeightAssignForEach(Func<double> func)
            {
                for (int i = 0; i < weights.LongLength; i++)
                    for (int j = 0; j < weights[i].matrix.GetLongLength(0); j++)
                        for (int k = 0; k < weights[i].matrix.GetLongLength(1); k++)
                            weights[i].matrix[j, k] = func();
            }

            public void BiasAssignForEach(Func<double> func)
            {
                for (int i = 0; i < layers.LongLength; i++)
                    for (int j = 0; j < layers[i].biases.LongLength; j++)
                        layers[i].biases[j] = func();
            }

            /// <summary>
            /// Backpropagates and updates weights, biases
            /// </summary>
            public void GradientDescent(double[] sampleOutputs, DenseNNForwardResult forwardLog, double learningRate)
            {
                double[] errors = new double[sampleOutputs.LongLength];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int i = 0; i < sampleOutputs.LongLength; i++)
                    errors[i] = (forwardLog.outputs[i] - sampleOutputs[i]) * layers[layers.LongLength - 1].FunctionDifferential(forwardLog.layerInputs[layers.LongLength - 1][i]);

                for (int i = 0; i < layers[layers.LongLength - 1].dim; i++)
                    layers[layers.LongLength - 1].biases[i] -= learningRate * errors[i]; // bias update

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
                        // weight update
                        weights[fromLayer - 1].matrix[i, j] -= learningRate * errors[i] * layers[fromLayer - 1].ForwardComp(forwardLog.layerInputs[fromLayer - 1][j]);

                        // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                        layerErrors[j] += errors[i] * weights[fromLayer - 1].matrix[i, j] * layers[fromLayer - 1].FunctionDifferential(forwardLog.layerInputs[fromLayer - 1][j]);
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
                double[][] layerInputs = new double[layers.LongLength][];
                double[] outputs = ForwardLayers(inputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs, outputs);
            }

            public double[] ForwardLayers(double[] inputs, int toLayer, int fromLayer, ref double[][] layerInputs)
            {
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
            public double[] outputs;

            public DenseNNForwardResult(double[][] layerInputs, double[] outputs) 
            {
                this.layerInputs = layerInputs;
                this.outputs = outputs;
            }
        }

        public class DenseNeuralNetworkBuilder : INeuralNetworkBuilder
        {
            public List<Layer> layers;

            public DenseNeuralNetworkBuilder() 
            {
                layers = new List<Layer>();
            }

            public void NewLayers(params Layer[] dims)
            {
                foreach (Layer dim in dims)
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

        public enum ActivationFunc
        {
            ReLU,
            Sigmoid,
            Tanh,
            Linear
        }

        public class ActivationLayer : Layer
        {
            public readonly ActivationFunc func;

            public ActivationLayer(int dim, ActivationFunc func) : base(dim)
            {
                this.func = func;
            }

            public override double ForwardComp(double x)
            {
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        return 1 / (1 + Math.Exp(-x));
                    case ActivationFunc.Tanh:
                        return Math.Tanh(x);
                    case ActivationFunc.ReLU:
                        return (x > 0) ? x : 0;
                    case ActivationFunc.Linear:
                    default: 
                        return x;
                }
            }

            public override double FunctionDifferential(double x)
            {
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        double sigmoid = ForwardComp(x);
                        return sigmoid * (1 - sigmoid);
                    case ActivationFunc.Tanh:
                        double sqrExp = Math.Exp(x);
                        sqrExp *= sqrExp;
                        return 4 / (sqrExp + (1 / sqrExp) + 2);
                    case ActivationFunc.ReLU:
                        return (x > 0) ? 1 : 0;
                    case ActivationFunc.Linear:
                    default: 
                        return 1;
                }
            }
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
                    yield return ForwardComp(inputs[i] + biases[i]);
            }

            public virtual double ForwardComp(double x) => x;

            /// <summary>
            /// Get <b>df(bias, x) / dx</b> such <b>x</b> can be another function
            /// </summary>
            public virtual double FunctionDifferential(double x) => 1;

            public static implicit operator Layer(int dim) => new Layer(dim);
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

                for (int i = 0; i < inputs.LongLength; i++)
                    output += matrix[outputIndex, i] * inputs[i];

                return output;
            }
        }
    }
}