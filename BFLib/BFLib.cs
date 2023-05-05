using System;
using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.IO;
using System.Data.Common;

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
            public void GradientDescent(double[,] sampleOutputs, DenseNNForwardResult forwardLog, double learningRate)
            {
                double[,] errors = new double[sampleOutputs.GetLength(0), sampleOutputs.GetLength(1)];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int sample = 0; sample < sampleOutputs.GetLength(0); sample++)
                    for (int i = 0; i < sampleOutputs.GetLength(1); i++)
                    {
                        errors[sample, i] = (forwardLog.outputs[sample, i] - sampleOutputs[sample, i]) * layers[layers.LongLength - 1].FunctionDifferential(forwardLog.layerInputs[layers.LongLength - 1][sample, i] + layers[layers.LongLength - 1].GetBias(i));
                    }

                GradientDescentLayers(errors, forwardLog, learningRate, layers.Length - 1);
            }

            /// <summary>
            /// <i>For recursion purpose only</i>. Backpropagates and updates specified layers
            /// </summary>
            /// <param name="errors">Error vector of the commencing layer (or <b>fromLayer</b>)</param>
            /// <param name="fromlayer"><i>For recursion purpose only</i>. Going backwards from the given <b>fromLayer</b> index</param>
            private void GradientDescentLayers(double[,] errors, DenseNNForwardResult forwardLog, double learningRate, int fromLayer)
            {
                if (fromLayer > 0)
                    GradientDescentLayers(
                        layers[fromLayer].GradientDescent(errors, forwardLog.layerInputs[fromLayer], layers[fromLayer - 1], forwardLog.layerInputs[fromLayer - 1], weights[fromLayer - 1], learningRate),
                        forwardLog,
                        learningRate,
                        fromLayer - 1
                        );
                else if (fromLayer == 0)
                    GradientDescentLayers(
                        layers[fromLayer].GradientDescent(errors, forwardLog.layerInputs[fromLayer], null, null, null, learningRate),
                        forwardLog,
                        learningRate,
                        fromLayer - 1
                        );
                else
                    return;
            }

            public DenseNNForwardResult Forward(double[] inputs)
            {
                double[,] wrapInputs = new double[1, inputs.Length];

                for (int i = 0; i < inputs.Length; i++)
                    wrapInputs[1, i] = inputs[i];

                double[][,] layerInputs = new double[layers.LongLength][,];
                double[,] outputs = ForwardLayers(wrapInputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs, outputs);
            }
            
            public DenseNNForwardResult Forward(double[,] inputs)
            {
                double[][,] layerInputs = new double[layers.LongLength][,];
                double[,] outputs = ForwardLayers(inputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs, outputs);
            }

            double[] ForwardLayers(double[] inputs, int toLayer, int fromLayer, ref double[][] layerInputs)
            {
                if (fromLayer < toLayer)
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                    layerInputs[toLayer] = inputs;

                return layers[toLayer].Forward(layerInputs[toLayer]);
            }

            double[,] ForwardLayers(double[,] inputs, int toLayer, int fromLayer, ref double[][,] layerInputs)
            {
                if (fromLayer < toLayer)
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                    layerInputs[toLayer] = inputs;

                return layers[toLayer].Forward(layerInputs[toLayer]);
            }
        }

        public class DenseNNForwardResult
        {
            public double[][,] layerInputs;
            public double[,] outputs;

            public DenseNNForwardResult(double[][,] layerInputs, double[,] outputs) 
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
            Linear,
            Custom
        }

        public class BatchNormLayer : ForwardLayer
        {
            public double gamma = 1, beta = 0;

            public BatchNormLayer(int dim, ForwardPort port) : base(dim, ActivationFunc.Custom, port, false) { }

            public override double[,] Forward(double[,] inputs)
            {
                int sampleSize = inputs.GetLength(0);
                double[,] result = new double[sampleSize, dim];

                for (int i = 0; i < dim; i++)
                {
                    double mean = 0, variance = 0;

                    for (int sample = 0; sample < result.GetLength(0); sample++)
                        mean += inputs[sample, i];
                    mean /= sampleSize;

                    for (int sample = 0; sample < result.GetLength(0); sample++)
                        variance += Math.Pow(inputs[sample, i] - mean, 2);
                    variance /= sampleSize;

                    for (int sample = 0; sample < result.GetLength(0); sample++)
                        result[sample, i] = gamma * Standardize(inputs[sample, i], mean, variance) + beta; 
                }

                return result;
            }

            public override double ForwardComp(double x)
            {
                return x * gamma + beta;
            }

            public override double[,] GradientDescent(double[,] errors, double[,] layerInputs, Layer? prevLayer, double[,]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
            {
                if (prevLayer == null)
                    return errors;

                int sampleSize = errors.GetLength(0);

                double[,] layerErrors = new double[sampleSize, prevLayer.dim];
                double[] means = new double[dim];
                double[] variances = new double[dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++)
                        means[i] += layerInputs[sample, i];
                    means[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variances[i] += Math.Pow(layerInputs[sample, i] - means[i], 2);
                    variances[i] /= sampleSize - 1;
                }

                double[] 
                    dbeta = new double[dim],
                    dgamma = new double[dim], 
                    dvariances = new double[dim],
                    dmeans = new double[dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++) 
                    {
                        dbeta[i] += errors[sample, i];
                        dgamma[i] += errors[sample, i] * Standardize(layerInputs[sample, i], means[i], variances[i]);

                        dvariances[i] += errors[sample, i] * (layerInputs[sample, i] - means[i]);
                        dmeans[i] += errors[sample, i];
                    }

                    dvariances[i] *= (-0.5d) * gamma * Math.Pow(variances[i], -1.5d);

                    dmeans[i] *= (gamma * sampleSize) / (Math.Sqrt(variances[i]) * dvariances[i] * 2);
                    // dmeans[i] = (-gamma) / Math.Sqrt(variances[i]); 
                    // dmeans[i] /= dvariances[i] * (-2) * (1 / sampleSize); 

                    for (int sample = 0; sample < sampleSize; sample++)
                        dmeans[i] += layerInputs[sample, i] - means[i];
                    dmeans[i] *= dvariances[i] * (-2);
                    dmeans[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        layerErrors[sample, i] =
                            (errors[sample, i] * gamma) / Math.Sqrt(variances[i]) +
                            dmeans[i] / sampleSize +
                            (2 * dvariances[i] * (layerInputs[sample, i] - means[i])) / sampleSize;
                }

                for (int i = 0; i < dim; i++)
                {
                    gamma -= dgamma[i] * learningRate;
                    beta -= dbeta[i] * learningRate;
                }

                if (port == ForwardPort.In)
                    return layerErrors;

                double[,] weightErrors = new double[errors.GetLength(0), prevLayer.dim];

                prevWeights.AssignForEach((inIndex, outIndex, weightValue) =>
                {
                    double weightErrorSum = 0;

                    // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                    for (int sample = 0; sample < prevLayerInputs.GetLength(0); sample++)
                    {
                        weightErrors[sample, inIndex] += errors[sample, outIndex] * weightValue * prevLayer.FunctionDifferential(prevLayerInputs[sample, inIndex] + prevLayer.GetBias(inIndex));
                        weightErrorSum += layerErrors[sample, outIndex] * prevLayer.ForwardComp(prevLayerInputs[sample, inIndex] + prevLayer.GetBias(inIndex));
                    }

                    // weight update
                    return weightValue - learningRate * weightErrorSum;
                });

                return weightErrors;
            }

            public static double Standardize(double x, double mean, double variance, double zeroSub = 0.00000001) => variance != 0 ? (x - mean) / Math.Sqrt(variance) : (x - mean) / zeroSub;

        }

        public class NormalizationLayer : ForwardLayer
        {
            public double gamma, beta;

            public NormalizationLayer(int dim, double min, double max, ForwardPort port) : base(dim, ActivationFunc.Custom, port, false)
            {
                gamma = 1 / (max - min);
                beta = -min;
            }

            public override double[,] GradientDescent(double[,] errors, double[,] layerInputs, Layer? prevLayer, double[,]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
            {
                return errors;
            }

            public override double ForwardComp(double x)
            {
                return x * gamma + beta;
            }
        }

        public class ForwardLayer : ActivationLayer
        {
            public enum ForwardPort
            {
                In,
                Out,
                Both
            }

            public readonly ForwardPort port;

            public ForwardLayer(int dim, ActivationFunc func, ForwardPort port, bool useBias = true) : base(dim, func, useBias) 
            {
                this.port = port;
            }

            public override IWeightMatrix GenerateWeightMatrix(Layer prevLayer)
            {
                if (port != ForwardPort.Out)
                {
                    if (prevLayer.dim == dim)
                        return new ForwardWeightMatrix(dim, useBias);
                    else
                        throw new Exception("Port port In");
                }
                else
                    return new DenseWeightMatrix(prevLayer.dim, dim);
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

            /// <returns>Returns descended errors</returns>
            public virtual double[,] GradientDescent(double[,] errors, double[,] layerInputs, Layer? prevLayer, double[,]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
            {
                for (int i = 0; i < dim; i++)
                {
                    // bias update
                    if (useBias)
                        for (int sample = 0; sample < errors.GetLength(0); sample++)
                            SetBias(i, GetBias(i) - learningRate * errors[sample, i]);
                }

                if (prevWeights == null)
                    return errors;

                double[,] layerErrors = new double[errors.GetLength(0), prevLayer.dim];

                prevWeights.AssignForEach((inIndex, outIndex, weightValue) =>
                {
                    double weightErrorSum = 0;

                    // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                    for (int sample = 0; sample < prevLayerInputs.GetLength(0); sample++)
                    {
                        layerErrors[sample, inIndex] += errors[sample, outIndex] * weightValue * prevLayer.FunctionDifferential(prevLayerInputs[sample, inIndex] + prevLayer.GetBias(inIndex));
                        weightErrorSum += errors[sample, outIndex] * prevLayer.ForwardComp(prevLayerInputs[sample, inIndex] + prevLayer.GetBias(inIndex));
                    }

                    // weight update
                    return weightValue - learningRate * weightErrorSum;
                });

                return layerErrors;
            }

            public virtual double[,] Forward(double[,] inputs)
            {
                double[,] result = new double[inputs.GetLength(0), dim];

                for (int i = 0; i < inputs.GetLength(0); i++)
                    for (int j = 0; j < dim; j++)
                        result[i, j] = ForwardComp(inputs[i, j]);

                return result;
            }

            public virtual double[] Forward(double[] inputs)
            {
                double[] result = new double[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = ForwardComp(inputs[i]);

                return result;
            }

            public virtual double ForwardComp(double x) => x;

            /// <summary>
            /// Get <b>df(bias, x) / dx</b> such <b>x</b> can be another function
            /// </summary>
            public virtual double FunctionDifferential(double x) => 1;

            public virtual IWeightMatrix GenerateWeightMatrix(Layer prevLayer)
            {
                if (prevLayer is ForwardLayer)
                    if (((ForwardLayer)prevLayer).port != ForwardLayer.ForwardPort.In)
                    {
                        if (prevLayer.dim == dim)
                            return new ForwardWeightMatrix(dim, prevLayer.useBias);
                        else
                            throw new Exception("Port port out");
                    }

                return new DenseWeightMatrix(prevLayer.dim, dim);
            }

            public static implicit operator Layer(int dim) => new Layer(dim);
        }

        public interface IWeightMatrix
        {
            public int inDim { get; }
            public int outDim { get; }

            public abstract double[] Forward(double[] inputs);

            public abstract double[,] Forward(double[,] inputs);

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

            public double[] Forward(double[] inputs)
            {
                double[] result = new double[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = ForwardComp(inputs, i);

                return result;
            }
            
            public double[,] Forward(double[,] inputs)
            {
                double[,] result = new double[inputs.GetLength(0), dim];

                for (int i = 0; i < inputs.GetLength(0); i++)
                    for (int j = 0; j < dim; j++)
                    {
                        if (useWeights)
                            result[i, j] = inputs[i, j] * matrix[j];
                        else
                            result[i, j] = inputs[i, j];
                    }

                return result;
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

            public double[] Forward(double[] inputs)
            {
                double[] result = new double[outDim];

                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        result[i] += inputs[j] * matrix[i, j];

                return result;
            }

            public double[,] Forward(double[,] inputs)
            {
                double[,] result = new double[inputs.GetLength(0), outDim];

                for (int i = 0; i < inputs.GetLength(0); i++)
                    for (int j = 0; j < outDim; j++)
                        for (int k = 0; k < inDim; k++)
                            result[i, j] += inputs[i, k] * matrix[j, k];

                return result;
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

    namespace Data
    {
        public static class UData
        {
            public object[][] RetrieveUDataFromCSV(string path, UDataInfo info, out string[] categories)
            {
                List<object[]> data = new List<object[]>();

                using (StreamReader reader = new StreamReader(path))
                {
                    categories = reader.ReadLine().Split(',');

                    if (info.types.Length != categories.Length)
                        throw new Exception("type info unmatch");

                    int distinctNum = 0;
                    foreach (DataType type in info.types)
                        if (type == DataType.DistinctInt)
                            distinctNum++;

                    List<string>[] distinctMatches = new List<string>[distinctNum];

                    while (!reader.EndOfStream)
                    {
                        string[] rawDataLine = reader.ReadLine().Split(',');
                        object[] dataLine = new object[rawDataLine.Length];

                        int curDistinct = 0;

                        for(int i = 0; i < rawDataLine.Length; i++)
                        {
                            if (string.IsNullOrEmpty(rawDataLine[i]))
                                continue;

                            switch (info.types[i])
                            {
                                case DataType.Int:
                                    dataLine[i] = int.Parse(rawDataLine[i]); 
                                    break;
                                case DataType.Double:
                                    dataLine[i] = double.Parse(rawDataLine[i]); 
                                    break;
                                case DataType.DistinctInt:
                                    dataLine[i] = double.Parse(rawDataLine[i]); 
                                    break;
                                case DataType.String:
                                default:
                                    dataLine[i] = rawDataLine[i];
                                    break;
                            }
                        }
                    }
                }
            }

            public string[] GetNotableDifferences(string[] source)
            {

                for (int i = 0; i < source.Length; i++)
                {

                }
            }

            public int GetOrAddDistinctIndex(List<string> source, string target)
            {
                for (int i = 0; i < source.Count; i++)
                {
                    if (source[i] == target) return i;
                }
            }
        }

        public enum DataType
        {
            String,
            Int,
            Double,
            DistinctInt
        }

        public struct UDataInfo
        {
            public DataType[] types;
            public DistinctIntDataInfo[] distinctData;

            public UDataInfo(DistinctIntDataInfo[] distinctData, params DataType[] types)
            {
                this.types = types;
                this.distinctData = distinctData;
            }

            public UDataInfo(params DataType[] types)
            {
                this.types = types;
                this.distinctData = new DistinctIntDataInfo[0];
            }
        }

        public struct DistinctIntDataInfo
        {
            public string category;
            public string[] encodings;

            public DistinctIntDataInfo(string category, string[] encodings)
            {
                this.category = category;
                this.encodings = encodings;
            }
        }
    }
}