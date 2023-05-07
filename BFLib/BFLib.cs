using System;
using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.IO;
using System.Data.Common;
using System.Net;
using System.Dynamic;
using System.Diagnostics.CodeAnalysis;
using System.Reflection.Emit;

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
            public void GradientDescent(double[][] sampleOutputs, DenseNNForwardResult forwardLog, double learningRate)
            {
                double[][] errors = new double[sampleOutputs.Length][];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int sample = 0; sample < sampleOutputs.Length; sample++)
                {
                    errors[sample] = new double[sampleOutputs[0].Length];

                    for (int i = 0; i < sampleOutputs[0].Length; i++)
                    {
                        errors[sample][i] = (forwardLog.outputs[sample][i] - sampleOutputs[sample][i]) * layers[layers.LongLength - 1].FunctionDifferential(forwardLog.layerInputs[layers.LongLength - 1][sample][i] + layers[layers.LongLength - 1].GetBias(i));
                    }
                }

                GradientDescentLayers(errors, forwardLog, learningRate, layers.Length - 1);
            }

            /// <summary>
            /// <i>For recursion purpose only</i>. Backpropagates and updates specified layers
            /// </summary>
            /// <param name="errors">Error vector of the commencing layer (or <b>fromLayer</b>)</param>
            /// <param name="fromlayer"><i>For recursion purpose only</i>. Going backwards from the given <b>fromLayer</b> index</param>
            private void GradientDescentLayers(double[][] errors, DenseNNForwardResult forwardLog, double learningRate, int fromLayer)
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
                double[][] wrapInputs = new double[1][];
                wrapInputs[0] = inputs;

                double[][][] layerInputs = new double[layers.LongLength][][];
                double[][] outputs = ForwardLayers(wrapInputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs, outputs);
            }
            
            public DenseNNForwardResult Forward(Dictionary<string, double>[] inputs)
            {
                double[][][] layerInputs = new double[layers.LongLength][][];
                double[][] outputs;

                if (layers[0] is InterfaceLayer)
                {
                    InterfaceLayer inputLayer = (InterfaceLayer)layers[0];
                    outputs = ForwardLayers(inputLayer.ToIndexInputs(inputs), layers.Length - 1, 0, ref layerInputs);
                }
                else
                    outputs = ForwardLayers(ToDoubleArrays(inputs), layers.Length - 1, 0, ref layerInputs);

                return new DenseNNForwardResult(layerInputs, outputs);
            }

            public DenseNNForwardResult Forward(double[][] inputs)
            {
                double[][][] layerInputs = new double[layers.LongLength][][];
                double[][] outputs = ForwardLayers(inputs, layers.Length - 1, 0, ref layerInputs);

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

            double[][] ForwardLayers(double[][] inputs, int toLayer, int fromLayer, ref double[][][] layerInputs)
            {
                if (fromLayer < toLayer)
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                    layerInputs[toLayer] = inputs;

                return layers[toLayer].Forward(layerInputs[toLayer]);
            }

            static double[][] ToDoubleArrays(Dictionary<string, double>[] content)
            {
                List<string> keyList = new List<string>();
                foreach (string label in content[0].Keys)
                    keyList.Add(label);

                double[][] result = new double[content.Length][];
                string[] keys = keyList.ToArray();

                for (int sample = 0; sample < content.Length; sample++)
                {
                    result[sample] = new double[keys.Length];
                    for (int i = 0; i < keys.Length; i++)
                        result[sample][i] = content[sample][keys[i]];
                }

                return result;
            }

        }

        public class DenseNNForwardResult
        {
            public double[][][] layerInputs;
            public double[][] outputs;

            public DenseNNForwardResult(double[][][] layerInputs, double[][] outputs) 
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

            public override double[][] Forward(double[][] inputs)
            {
                int sampleSize = inputs.Length;
                double[][] result = new double[sampleSize][];

                for (int sample = 0; sample < result.Length; sample++)
                    result[sample] = new double[dim];

                for (int i = 0; i < dim; i++)
                {
                    double mean = 0, variance = 0;

                    for (int sample = 0; sample < result.Length; sample++)
                        mean += inputs[sample][i];
                    mean /= sampleSize;

                    for (int sample = 0; sample < result.Length; sample++)
                        variance += Math.Pow(inputs[sample][i] - mean, 2);
                    variance /= sampleSize;

                    for (int sample = 0; sample < result.Length; sample++)
                        result[sample][i] = gamma * Standardize(inputs[sample][i], mean, variance) + beta; 
                }

                return result;
            }

            public override double ForwardComp(double x)
            {
                return x * gamma + beta;
            }

            public override double[][] GradientDescent(double[][] errors, double[][] layerInputs, Layer? prevLayer, double[][]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
            {
                if (prevLayer == null)
                    return errors;

                int sampleSize = errors.Length;

                double[][] layerErrors = new double[sampleSize][];
                double[] means = new double[dim];
                double[] variances = new double[dim];

                for (int sample = 0; sample < sampleSize; sample++)
                    layerErrors[sample] = new double[prevLayer.dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++)
                        means[i] += layerInputs[sample][i];
                    means[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variances[i] += Math.Pow(layerInputs[sample][i] - means[i], 2);
                    variances[i] /= sampleSize - 1;
                    variances[i] += 0.000001d;
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
                        dbeta[i] += errors[sample][i];
                        dgamma[i] += errors[sample][i] * Standardize(layerInputs[sample][i], means[i], variances[i]);

                        dvariances[i] += errors[sample][i] * (layerInputs[sample][i] - means[i]);
                        dmeans[i] += errors[sample][i];
                    }

                    dvariances[i] *= (-0.5d) * gamma * Math.Pow(variances[i], -1.5d);
                    dvariances[i] += 0.000001d;

                    dmeans[i] *= (gamma * sampleSize) / (Math.Sqrt(variances[i]) * dvariances[i] * 2);
                    // dmeans[i] = (-gamma) / Math.Sqrt(variances[i]); 
                    // dmeans[i] /= dvariances[i] * (-2) * (1 / sampleSize); 

                    for (int sample = 0; sample < sampleSize; sample++)
                        dmeans[i] += layerInputs[sample][i] - means[i];
                    dmeans[i] *= dvariances[i] * (-2);
                    dmeans[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        layerErrors[sample][i] =
                            (errors[sample][i] * gamma) / Math.Sqrt(variances[i]) +
                            dmeans[i] / sampleSize +
                            (2 * dvariances[i] * (layerInputs[sample][i] - means[i])) / sampleSize;
                }

                for (int i = 0; i < dim; i++)
                {
                    gamma -= dgamma[i] * learningRate;
                    beta -= dbeta[i] * learningRate;
                }

                if (port == ForwardPort.In)
                    return layerErrors;

                double[][] weightErrors = new double[errors.Length][];
                for (int i = 0; i < errors.Length; i++)
                    weightErrors[i] = new double[prevLayer.dim];

                prevWeights.AssignForEach((inIndex, outIndex, weightValue) =>
                {
                    double weightErrorSum = 0;

                    // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                    for (int sample = 0; sample < prevLayerInputs.Length; sample++)
                    {
                        weightErrors[sample][inIndex] += errors[sample][outIndex] * weightValue * prevLayer.FunctionDifferential(prevLayerInputs[sample][inIndex] + prevLayer.GetBias(inIndex));
                        weightErrorSum += layerErrors[sample][outIndex] * prevLayer.ForwardComp(prevLayerInputs[sample][inIndex] + prevLayer.GetBias(inIndex));
                    }

                    // weight update
                    return weightValue - learningRate * weightErrorSum;
                });

                return weightErrors;
            }

            public static double Standardize(double x, double mean, double variance, double zeroSub = 0.000001) => variance != 0 ? (x - mean) / Math.Sqrt(variance) : (x - mean) / zeroSub;

        }

        public class NormalizationLayer : ForwardLayer
        {
            public double gamma, beta;

            public NormalizationLayer(int dim, double min, double max, ForwardPort port) : base(dim, ActivationFunc.Custom, port, false)
            {
                gamma = 1 / (max - min);
                beta = -min;
            }

            public override double[][] GradientDescent(double[][] errors, double[][] layerInputs, Layer? prevLayer, double[][]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
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

        public enum InterfaceLayerType
        {
            Input,
            Output
        }

        public class InterfaceLayer : ActivationLayer
        {
            public readonly string[] labels;
            public readonly InterfaceLayerType type;

            public InterfaceLayer (string[] labels, InterfaceLayerType type, ActivationFunc func = ActivationFunc.Linear, bool useBias = true) : base(labels.Length, func, useBias)
            {
                this.labels = labels;
                this.type = type;
            }

            public double[][] ToIndexInputs(Dictionary<string, double>[] inputs)
            {
                double[][] result = new double[inputs.Length][];
                string[] labels = inputs[0].Keys.ToArray();

                for (int sample = 0; sample < inputs.Length; sample++)
                {
                    result[sample] = new double[labels.Length];
                    for(int i = 0; i < labels.Length; i++)
                        result[sample][i] = inputs[sample][labels[i]];
                }

                return result;
            }

            public double[][] ToIndexInputs(Dictionary<string, double>[] inputs, Dictionary<string, Tuple<double, double>> ranges)
            {
                double[][] result = new double[inputs.Length][];
                string[] labels = inputs[0].Keys.ToArray();

                for (int sample = 0; sample < inputs.Length; sample++)
                {
                    result[sample] = new double[labels.Length];
                    for (int i = 0; i < labels.Length; i++)
                    {
                        Tuple<double, double> range = ranges[labels[i]];

                        if (range != null)
                            result[sample][i] = (inputs[sample][labels[i]] - range.Item1) / (range.Item2 - range.Item1);
                        else
                            result[sample][i] = inputs[sample][labels[i]];
                    }
                }

                return result;
            }

            public virtual double[][] Forward(Dictionary<string, double>[] inputs)
            {
                return Forward(ToIndexInputs(inputs));
            }

            public virtual double[][] Forward(Dictionary<string, double>[] inputs, Dictionary<string, Tuple<double, double>> ranges)
            {
                return Forward(ToIndexInputs(inputs, ranges));
            }

            public static double[][] OrderByASCII(Dictionary<string, double>[] content)
            {
                double[][] result = new double[content.Length][];
                List<string> labels = new List<string>();

                foreach(string label in content[0].Keys)
                    labels.Add(label);

                string[] sortLabels = labels.ToArray();

                QuickSort(ref sortLabels);

                for (int sample = 0; sample < content.Length; sample++)
                {
                    result[sample] = new double[sortLabels.Length];
                    for (int iLabel = 0; iLabel < sortLabels.Length; iLabel++)
                        result[sample][iLabel] = content[sample][sortLabels[iLabel]];
                }

                return result;
            }

            public static Dictionary<string, double>[] DicOrderByASCII(Dictionary<string, double>[] content)
            {
                Dictionary<string, double>[] result = new Dictionary<string, double>[content.Length];
                List<string> labels = new List<string>();

                foreach(string label in content[0].Keys)
                    labels.Add(label);

                string[] sortLabels = labels.ToArray();

                QuickSort(ref sortLabels);

                for (int sample = 0; sample < content.Length; sample++)
                {
                    result[sample] = new Dictionary<string, double>();
                    foreach (string label in sortLabels)
                        result[sample].Add(label, content[sample][label]);
                }

                return result;
            }

            static void QuickSort(ref string[] content, int lIndex = 0, int rIndex = -1, int charIndex = 0, bool doneSort = true) 
            {
                if (lIndex == rIndex)
                    return;

                int lIterate = lIndex,
                    rIterate = (rIndex == -1) ? content.Length - 1 : rIndex;
                int pivot = lIndex + (rIterate - lIterate + 1) / 2;
                bool swap = false;

                while (lIterate != pivot || rIterate != pivot)
                {
                    while (lIterate < pivot)
                    {
                        if (content[lIterate].Length > charIndex &&
                            (content[pivot].Length <= charIndex || content[lIterate][charIndex] > content[pivot][charIndex]))
                        {
                            if (!swap)
                                swap = true;
                            else if (rIterate != pivot)
                            {
                                Swap(lIterate, rIterate, ref content);
                                swap = false;
                                rIterate--;
                                lIterate++;
                            }
                            else
                            {
                                Swap(lIterate, pivot, ref content);
                                swap = false;
                                pivot = lIterate;
                            }

                            break;
                        }
                        lIterate++;
                    }

                    while (rIterate > pivot)
                    {
                        if (content[pivot].Length > charIndex &&
                            (content[rIterate].Length <= charIndex || content[rIterate][charIndex] < content[pivot][charIndex]))
                        {
                            if (!swap)
                                swap = true;
                            else if (lIterate != pivot)
                            {
                                Swap(lIterate, rIterate, ref content);
                                swap = false;
                                lIterate++;
                                rIterate--;
                            }
                            else
                            {
                                Swap(pivot, rIterate, ref content);
                                swap = false;
                                pivot = rIterate;
                            }

                            break;
                        }
                        rIterate--;
                    }
                }

                int rBound = (rIndex == -1) ? content.Length - 1 : rIndex;

                if (pivot != rBound) QuickSort(ref content, pivot + 1, rBound, charIndex, false);
                if (pivot != lIndex) QuickSort(ref content, lIndex, pivot - 1, charIndex, false);

                if (!doneSort)
                    return;

                int compareIndex = lIndex;
                bool isExceededAll = false;
                for (int i = lIndex + 1; i < rBound + 1; i++)
                {
                    if (content[i].Length > charIndex && content[compareIndex].Length > charIndex)
                    {
                        if (content[compareIndex][charIndex] != content[i][charIndex])
                        {
                            QuickSort(ref content, compareIndex, i - 1, charIndex + 1, true);
                            compareIndex = i;
                        }
                    }
                    else
                        isExceededAll = true;
                }

                if (!isExceededAll && compareIndex != lIndex)
                    QuickSort(ref content, compareIndex, rBound, charIndex + 1, true);
            }

            static void Swap(int a, int b, ref string[] content) 
            {
                string temp = content[a];
                content[a] = content[b];
                content[b] = temp;
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
            public virtual double[][] GradientDescent(double[][] errors, double[][] layerInputs, Layer? prevLayer, double[][]? prevLayerInputs, IWeightMatrix? prevWeights, double learningRate)
            {
                for (int i = 0; i < dim; i++)
                {
                    // bias update
                    if (useBias)
                        for (int sample = 0; sample < errors.Length; sample++)
                            SetBias(i, GetBias(i) - learningRate * errors[sample][i]);
                }

                if (prevWeights == null)
                    return errors;

                double[][] layerErrors = new double[errors.Length][];

                for (int i = 0; i < errors.Length; i++)
                    layerErrors[i] = new double[prevLayer.dim];

                prevWeights.AssignForEach((inIndex, outIndex, weightValue) =>
                {
                    double weightErrorSum = 0;

                    // pass-down-error = error * (corresponding weight) * (derivative of (l - 1) layer function with respect to its input)
                    for (int sample = 0; sample < prevLayerInputs.Length; sample++)
                    {
                        layerErrors[sample][inIndex] += errors[sample][outIndex] * weightValue * prevLayer.FunctionDifferential(prevLayerInputs[sample][inIndex] + prevLayer.GetBias(inIndex));
                        weightErrorSum += errors[sample][outIndex] * prevLayer.ForwardComp(prevLayerInputs[sample][inIndex] + prevLayer.GetBias(inIndex));
                    }

                    // weight update
                    return weightValue - learningRate * weightErrorSum;
                });

                return layerErrors;
            }

            public virtual double[][] Forward(double[][] inputs)
            {
                double[][] result = new double[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new double[dim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < dim; j++)
                        result[i][j] = ForwardComp(inputs[i][j]);

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

            public abstract double[][] Forward(double[][] inputs);

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
            
            public double[][] Forward(double[][] inputs)
            {
                double[][] result = new double[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new double[dim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        if (useWeights)
                            result[i][j] = inputs[i][j] * matrix[j];
                        else
                            result[i][j] = inputs[i][j];
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

            public double[][] Forward(double[][] inputs)
            {
                double[][] result = new double[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new double[outDim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < outDim; j++)
                        for (int k = 0; k < inDim; k++)
                            result[i][j] += inputs[i][k] * matrix[j, k];

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
            public static string[] GetCategoriesFromCSV(string path)
            {
                string[] cats;
                using (StreamReader reader = new StreamReader(path))
                    cats = reader.ReadLine().Split(',');

                return cats;
            }

            public static Dictionary<string, double>[] RetrieveDistinctIntDataFromCSV(string path, int retrieveAmount, params string[] retrieveCats)
            {
                string[] cats = GetCategoriesFromCSV(path);
                List<string> neglectCats = new List<string>();
                DistinctIntDataInfo[] encodings;
                Dictionary<string, Tuple<double, double>> ranges;

                DataType[] dataTypes = new DataType[cats.Length];

                for (int i = 0; i < cats.Length; i++)
                {
                    bool going = true;
                    foreach(string cat in retrieveCats)
                        if (cats[i] == cat)
                        {
                            dataTypes[i] = DataType.DistinctInt;
                            going = false;
                            break;
                        }

                    if (!going)
                        continue;

                    dataTypes[i] = DataType.Neglect;
                }

                for (int i = 0; i < dataTypes.Length; i++)
                    if (dataTypes[i] == DataType.Neglect)
                        neglectCats.Add(cats[i]);

                UDataInfo info = new UDataInfo(neglectCats.ToArray(), dataTypes);

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out ranges, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveNumberDataFromCSV(string path, int retrieveAmount, out Dictionary<string, Tuple<double, double>> ranges, params string[] retrieveCats)
            {
                string[] cats = GetCategoriesFromCSV(path);
                List<string> neglectCats = new List<string>();
                DistinctIntDataInfo[] encodings;

                DataType[] dataTypes = new DataType[cats.Length];

                for (int i = 0; i < cats.Length; i++)
                {
                    bool going = true;
                    foreach(string cat in retrieveCats)
                        if (cats[i] == cat)
                        {
                            dataTypes[i] = DataType.Double;
                            going = false;
                            break;
                        }

                    if (!going)
                        continue;

                    dataTypes[i] = DataType.Neglect;
                }

                for (int i = 0; i < dataTypes.Length; i++)
                    if (dataTypes[i] == DataType.Neglect)
                        neglectCats.Add(cats[i]);

                UDataInfo info = new UDataInfo(neglectCats.ToArray(), dataTypes);

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out ranges, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                Dictionary<string, Tuple<double, double>> ranges;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out ranges, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out Dictionary<string, Tuple<double, double>> ranges, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out ranges, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, Tuple<double, double>> ranges, int retrieveAmount = -1)
            {
                string[] cats;
                return RetrieveUDataFromCSV(path, info, out cats, out distinctEncodings, out ranges, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out string[] categories, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, Tuple<double, double>> ranges, int retrieveAmount = -1)
            {
                List<Dictionary<string, double>> data = new List<Dictionary<string, double>>();
                List<double[]> rawData = new List<double[]>();
                ranges = new Dictionary<string, Tuple<double, double>>();

                using (StreamReader reader = new StreamReader(path))
                {
                    categories = reader.ReadLine().Split(',');

                    if (info.types.Length != categories.Length)
                        throw new Exception("type info unmatch");

                    List<int> iteratingIndexList = new List<int>();

                    for (int i = 0; i < categories.Length; i++)
                    {
                        int j = 0;
                        for (; j < info.neglectCats.Length; j++)
                            if (categories[i] == info.neglectCats[j])
                                break;

                        if (j == info.neglectCats.Length)
                            iteratingIndexList.Add(i);
                    }

                    int[] iteratingIndices = iteratingIndexList.ToArray();

                    foreach (int i in iteratingIndices)
                    {
                        if (info.types[i] == DataType.Double)
                            ranges.Add(categories[i], new Tuple<double, double>(double.NaN, double.NaN));
                    }

                    Dictionary<int, int> givenDistinctDataIndices = new Dictionary<int, int>();

                    for (int i = 0; i < info.distinctData.Length; i++)
                        for (int j = 0; j < categories.Length; j++)
                            if (info.distinctData[i].category == categories[j])
                            {
                                givenDistinctDataIndices.Add(j, i);
                                break;
                            }

                    List<int> scoutDistinctDataIndices = new List<int>();
                    for (int i = 0; i < info.types.Length; i++)
                        if (info.types[i] == DataType.DistinctInt)
                            if (!givenDistinctDataIndices.Keys.Contains(i))
                                scoutDistinctDataIndices.Add(i);

                    List<string>[] scoutDistinctData = new List<string>[scoutDistinctDataIndices.Count];

                    for (int i = 0; i < scoutDistinctData.Length; i++)
                        scoutDistinctData[i] = new List<string>();

                    int retrieveCount = 0;
                    while (!reader.EndOfStream && (retrieveCount < retrieveAmount || retrieveAmount == -1))
                    {
                        string[] rawDataLine = reader.ReadLine().Split(',');
                        double[] dataLine = new double[iteratingIndices.Length];

                        int curDistinct = 0;

                        for (int i = 0; i < iteratingIndices.Length; i++)
                        {
                            if (string.IsNullOrEmpty(rawDataLine[iteratingIndices[i]]))
                                continue;

                            bool added = false;
                            foreach (int index in givenDistinctDataIndices.Keys)
                                if (iteratingIndices[i] == index)
                                {
                                    for (int j = 0; j < info.distinctData[givenDistinctDataIndices[index]].encodings.Length; j++)
                                        if (info.distinctData[givenDistinctDataIndices[index]].encodings[j] == rawDataLine[iteratingIndices[i]])
                                            dataLine[iteratingIndices[i]] = j;

                                    added = true;
                                }

                            if (added)
                                continue;

                            switch (info.types[iteratingIndices[i]])
                            {
                                case DataType.Double:
                                    dataLine[i] = double.Parse(rawDataLine[iteratingIndices[i]]);

                                    double min = ranges[categories[iteratingIndices[i]]].Item1;
                                    double max = ranges[categories[iteratingIndices[i]]].Item2;

                                    if (min > dataLine[i] || double.IsNaN(min)) min = dataLine[i];
                                    if (max < dataLine[i] || double.IsNaN(max)) max = dataLine[i];

                                    ranges[categories[iteratingIndices[i]]] = new Tuple<double, double>(min, max);

                                    break;
                                case DataType.DistinctInt:
                                    int j = 0;
                                    for (; j < scoutDistinctData[curDistinct].Count;)
                                    {
                                        if (scoutDistinctData[curDistinct][j] == rawDataLine[iteratingIndices[i]])
                                        {
                                            dataLine[i] = j;
                                            break;
                                        }
                                        j++;
                                    }

                                    if (j != scoutDistinctData[curDistinct].Count)
                                    {
                                        curDistinct++;
                                        break;
                                    }

                                    scoutDistinctData[curDistinct].Add(rawDataLine[iteratingIndices[i]]);
                                    dataLine[i] = scoutDistinctData[curDistinct].Count - 1;
                                    curDistinct++;
                                    break;
                            }
                        }

                        rawData.Add(dataLine);
                        retrieveCount++;
                    }

                    DistinctIntDataInfo[] distinctInfos = new DistinctIntDataInfo[scoutDistinctData.Length + info.distinctData.Length];

                    for (int i = 0; i < scoutDistinctData.Length; i++)
                        distinctInfos[i] = new DistinctIntDataInfo(categories[scoutDistinctDataIndices[i]], scoutDistinctData[i].ToArray());

                    for (int i = scoutDistinctData.Length; i < info.distinctData.Length; i++)
                        distinctInfos[i] = info.distinctData[i - scoutDistinctData.Length];

                    distinctEncodings = distinctInfos;

                    for (int sample = 0; sample < rawData.Count; sample++)
                    {
                        data.Add(new Dictionary<string, double>());

                        for (int i = 0; i < iteratingIndices.Length; i++)
                            data[sample].Add(categories[iteratingIndices[i]], rawData[sample][i]);
                    }
                }

                return data.ToArray();
            }
        }

        public enum DataType
        {
            Neglect,
            Double,
            DistinctInt
        }

        public struct UDataInfo
        {
            public DataType[] types;
            public DistinctIntDataInfo[] distinctData;
            public string[] neglectCats;

            public UDataInfo(string[] categories, params Tuple<string, DataType>[] catTypes)
            {
                types = new DataType[categories.Length];
                for (int i = 0; i < categories.Length; i++)
                {
                    types[i] = DataType.Neglect;
                    foreach(var catType in catTypes)
                        if(catType.Item1 == categories[i])
                        {
                            types[i] = catType.Item2;
                            break;
                        }
                }

                List<string> negCatList = new List<string>();
                for (int i = 0; i < categories.Length; i++)
                    if (types[i] == DataType.Neglect)
                        negCatList.Add(categories[i]);

                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = negCatList.ToArray();
            }

            public UDataInfo(string[] neglectCats, params DataType[] types)
            {
                this.types = types;
                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = neglectCats;
            }

            public UDataInfo(DistinctIntDataInfo[] distinctData, string[] neglectCats, params DataType[] types)
            {
                this.types = types;
                this.distinctData = distinctData;
                this.neglectCats = neglectCats;
            }

            public UDataInfo(DistinctIntDataInfo[] distinctData, params DataType[] types)
            {
                this.types = types;
                this.distinctData = distinctData;
                this.neglectCats = new string[0];
            }

            public UDataInfo(params DataType[] types)
            {
                this.types = types;
                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = new string[0];
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