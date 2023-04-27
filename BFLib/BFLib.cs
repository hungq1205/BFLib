using System;
using System.Buffers;
using System.Globalization;
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
            public readonly IWeightMatrix[] weights;

            public DenseNeuralNetwork(params Layer[] dims)
            {
                using (DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder())
                {
                    builder.NewLayers(dims);

                    Tuple<Layer[], IWeightMatrix[]> bundle = builder.Build();

                    this.layers = bundle.Item1;
                    this.weights = bundle.Item2;
                    this.inDim = layers[0].dim;
                    this.outDim = layers[layers.LongLength - 1].dim;
                }
            }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, bool disposeAfterwards = true)
            {
                Tuple<Layer[], IWeightMatrix[]> bundle = builder.Build();

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
                    weights[i].AssignForEach((inIndex, outIndex, weight) => func());
            }

            public void BiasAssignForEach(Func<double> func)
            {
                for (int i = 0; i < layers.LongLength; i++)
                    for (int j = 0; j < layers[i].dim; j++)
                        layers[i].SetBias(j, func());
            }

            /// <summary>
            /// Backpropagates and updates weights, biases
            /// </summary>
            public void GradientDescent(double[] sampleOutputs, DenseNNForwardResult forwardLog, double learningRate)
            {
                double[] errors = new double[sampleOutputs.LongLength];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int i = 0; i < sampleOutputs.LongLength; i++)
                    errors[i] = (forwardLog.outputs[i] - sampleOutputs[i]) * layers[layers.LongLength - 1].FunctionDifferential(forwardLog.layerInputs[layers.LongLength - 1][i] + layers[layers.LongLength - 1].GetBias(i));

                if (layers[layers.LongLength - 1].useBias)
                    for (int i = 0; i < layers[layers.LongLength - 1].dim; i++)
                        layers[layers.LongLength - 1].SetBias(i, layers[layers.LongLength - 1].GetBias(i) - learningRate * errors[i]); // bias update

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

                weights[fromLayer - 1].AssignForEach((inIndex, outIndex, weightValue) =>
                {
                    double biasTemp = layers[fromLayer - 1].GetBias(inIndex);

                    // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                    layerErrors[inIndex] += errors[outIndex] * weightValue * layers[fromLayer - 1].FunctionDifferential(forwardLog.layerInputs[fromLayer - 1][inIndex] + layers[fromLayer - 1].GetBias(inIndex));

                    // weight update
                    return weightValue - learningRate * errors[outIndex] * layers[fromLayer - 1].ForwardComp(forwardLog.layerInputs[fromLayer - 1][inIndex] + biasTemp);
                });

                for (int i = 0; i < layers[fromLayer - 1].dim; i++)
                {
                    // bias update
                    if (layers[fromLayer - 1].useBias)
                        layers[fromLayer - 1].SetBias(i, layers[fromLayer - 1].GetBias(i) - learningRate * layerErrors[i]);
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
                layers.Add(layer);
            }

            public Tuple<Layer[], IWeightMatrix[]> Build()
            {
                IWeightMatrix[] weights = new IWeightMatrix[layers.Count - 1];

                for (int i = 1; i < layers.Count; i++)
                    weights[i - 1] = layers[i].GenerateWeightMatrix(layers[i - 1]);

                return (layers.ToArray(), weights).ToTuple();
            }

            public void Dispose()
            {
                GC.SuppressFinalize(this);
            }
        }

        public interface INeuralNetworkBuilder : IDisposable
        {
            public abstract Tuple<Layer[], IWeightMatrix[]> Build();
        }

        public enum ActivationFunc
        {
            ReLU,
            Sigmoid,
            Tanh,
            NaturalLog,
            Exponential,
            Linear
        }

        public class ForwardLayer : ActivationLayer
        {
            public ForwardLayer(int dim, ActivationFunc func, bool useBias = true) : base(dim, func, useBias) { }

            public override IWeightMatrix GenerateWeightMatrix(Layer prevLayer)
            {
                if (prevLayer.dim == dim)
                    return new ForwardWeightMatrix(dim, useBias);

                return base.GenerateWeightMatrix(prevLayer);
            }
        }

        public class ActivationLayer : Layer
        {
            public readonly ActivationFunc func;

            public ActivationLayer(int dim, ActivationFunc func, bool useBias = true) : base(dim, useBias)
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
                    case ActivationFunc.NaturalLog:
                        return Math.Log(x);
                    case ActivationFunc.Exponential:
                        return Math.Exp(x);
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
                    case ActivationFunc.NaturalLog:
                        return 1 / x;
                    case ActivationFunc.Exponential:
                        return Math.Exp(x);
                    case ActivationFunc.Linear:
                    default: 
                        return 1;
                }
            }
        }

        public class Layer
        {
            public readonly int dim;
            public readonly bool useBias;

            double[] biases;

            public Layer(int dim, bool useBias = true)
            {
                this.dim = dim;
                this.useBias = useBias;
                this.biases = new double[dim];
            }
            
            public Layer(double[] biases)
            {
                this.dim = biases.Length;
                this.biases = biases;
            }

            public virtual double GetBias(int index) => useBias ? biases[index] : 0;

            public virtual void SetBias(int index, double value) => biases[index] = useBias ? value : 0;

            public virtual IEnumerable<double> Forward(double[] inputs)
            { 
                for (int i = 0; i < dim; i++)
                    yield return ForwardComp(inputs[i]);
            }

            public virtual double ForwardComp(double x) => x;

            /// <summary>
            /// Get <b>df(bias, x) / dx</b> such <b>x</b> can be another function
            /// </summary>
            public virtual double FunctionDifferential(double x) => 1;

            public virtual IWeightMatrix GenerateWeightMatrix(Layer prevLayer) => new DenseWeightMatrix(prevLayer.dim, dim);

            public static implicit operator Layer(int dim) => new Layer(dim);
        }

        public interface IWeightMatrix
        {
            public int inDim { get; }
            public int outDim { get; }

            public abstract IEnumerable<double> Forward(double[] inputs);

            public abstract double ForwardComp(double[] inputs, int outputIndex);

            public abstract void AssignForEach(Func<int, int, double, double> value);

            public abstract bool TrySetWeight(int inIndex, int outIndex, double value);

            public abstract bool TryGetWeight(int inIndex, int outIndex, out double weight);

            public abstract double GetWeight(int inIndex, int outIndex);
        }

        public class ForwardWeightMatrix : IWeightMatrix
        {
            public int inDim => dim;
            public int outDim => dim;

            public readonly int dim;

            public readonly bool useWeights;

            public double[] matrix;

            public ForwardWeightMatrix(int dim, bool useWeights = true)
            {
                this.dim = dim;
                this.useWeights = useWeights;
                this.matrix = new double[dim];
            }

            public void AssignForEach(Func<int, int, double, double> value)
            {
                for (int i = 0; i < dim; i++) {
                    if (useWeights)
                        matrix[i] = value(i, i, matrix[i]);
                    else
                        value(i, i, 1);
                }
            }

            public IEnumerable<double> Forward(double[] inputs)
            {
                for (int i = 0; i < dim; i++)
                    yield return ForwardComp(inputs, i);
            }

            public double ForwardComp(double[] inputs, int outputIndex)
            {
                if (useWeights)
                    return inputs[outputIndex] * matrix[outputIndex];
                else
                    return inputs[outputIndex];
            }

            public double GetWeight(int inIndex, int outIndex)
            {
                if (useWeights)
                {
                    if (inIndex == outIndex && inIndex < dim)
                        return matrix[inIndex];
                }
                else if (inIndex == outIndex)
                    return 1;
                else
                    return 0;

                throw new Exception("No weight here bro");
            }

            public bool TryGetWeight(int inIndex, int outIndex, out double weight)
            {
                if (useWeights)
                    weight = matrix[inIndex];
                else if (inIndex == outIndex)
                    weight = 1;
                else
                    weight = 0;

                return inIndex == outIndex && inIndex < dim;
            }

            public bool TrySetWeight(int inIndex, int outIndex, double value)
            {
                if (useWeights && inIndex == outIndex && inIndex < dim)
                {
                    matrix[inIndex] = value;
                    return true;
                }

                return false;
            }
        }

        public class DenseWeightMatrix : IWeightMatrix
        {
            public int inDim => _inDim;
            public int outDim => _outDim;

            int _inDim, _outDim;

            public double[,] matrix;

            public DenseWeightMatrix(int inDim, int outDim)
            {
                this._inDim = inDim;
                this._outDim = outDim;
                this.matrix = new double[outDim, inDim];
            }

            public DenseWeightMatrix(double[,] matrix)
            {
                this._inDim = matrix.GetLength(1);
                this._outDim = matrix.GetLength(0);

                this.matrix = matrix;             // need testing for shallow cloned or not
            }

            public bool TryGetWeight(int inIndex, int outIndex, out double weight)
            {
                weight = matrix[outIndex, inIndex];
                return true;
            }

            public double GetWeight(int inIndex, int outIndex) => matrix[outIndex, inIndex];

            public void AssignForEach(Func<int, int, double, double> value)
            {
                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        matrix[i, j] = value(j, i, matrix[i, j]);
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

            public bool TrySetWeight(int inIndex, int outIndex, double value)
            {
                matrix[outIndex, inIndex] = value; 
                return true;
            }
        }
    }
}