using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace BFLib
{
    namespace AI
    {
        public interface INeuralNetwork
        {
            public int inDim { get; }
            public int outDim { get; }
            public Optimizer optimizer { get; }
            public Layer[] layers { get; }
            public WeightMatrix[] weights { get; }

            public ForwardResult Forward(IForwardInput[] inputs);

            public void GradientDescent(double[][] sampleOutputs, ForwardResult forwardLog);

            /// <summary>
            /// </summary>
            /// <param name="func">Takes current weight as parameter and returns a new weight</param>
            public void WeightAssignForEach(Func<double, double> func);

            /// <summary>
            /// </summary>
            /// <param name="func">Takes current bias as parameter and returns a new bias</param>
            public void BiasAssignForEach(Func<double, double> func);
        }

        public abstract class ForwardResult
        {
            /// <summary>
            /// [sample][perceptron]
            /// </summary>
            public double[][] outputs;

            public ForwardResult(double[][] outputs)
            {
                this.outputs = outputs;
            }
        }

        public class RecurrentForwardResult : ForwardResult
        {
            /// <summary>
            /// [nodeIndex][layerIndex][sample][perceptron]
            /// </summary>
            public double[][][][] layerInputs;

            public RecurrentForwardResult(double[][][][] layerInputs, double[][] outputs) : base(outputs)
            {
                this.layerInputs = layerInputs;
            }
        }

        public class DenseForwardResult : ForwardResult
        {
            /// <summary>
            /// [layerIndex][sample][perceptron]
            /// </summary>
            public double[][][] layerInputs;

            public DenseForwardResult(double[][][] layerInputs, double[][] outputs) : base(outputs)
            {
                this.layerInputs = layerInputs;
            }
        }

        public interface IForwardInput { }

        public class DenseForwardInput : IForwardInput
        {
            /// <summary>
            /// [inputs]
            /// </summary>
            public double[] dense;

            public DenseForwardInput(double[] dense)
            {
                this.dense = dense;
            }
        }

        //public class RecurrentForwardInput : IForwardInput
        //{
        //    /// <summary>
        //    /// [nodeIndex][inputs]
        //    /// </summary>
        //    public double[][] recurrent;

        //    public RecurrentForwardInput(double[][] recurrent)
        //    {
        //        this.recurrent = recurrent;
        //    }
        //}

        //public class RecurrentNeuralNetwork : INeuralNetwork
        //{
        //    public int inDim { get; private set; }
        //    public int outDim { get; private set; }
        //    public Optimizer optimizer { get; private set; }
        //    public Layer[] layers { get; private set; }
        //    public WeightMatrix[] weights { get; private set; }

        //    public readonly int nodeCount, hiddenDim;
        //    public readonly ActivationFunc outFunc, hiddenFunc;
        //    public readonly double[] outBiases;
        //    public readonly WeightMatrix[] outWeights;

        //    public RecurrentNeuralNetwork(int nodeCount, int inDim, int outDim, int hiddenDim = 1, ActivationFunc outFunc = ActivationFunc.Linear, ActivationFunc hiddenFunc = ActivationFunc.Tanh) : base()
        //    {
        //        if (outDim > nodeCount)
        //            throw new Exception("No RNN for u");

        //        this.inDim = inDim;
        //        this.outDim = outDim;
        //        this.nodeCount = nodeCount;
        //        this.hiddenDim = hiddenDim;
        //        this.outFunc = outFunc;
        //        this.hiddenFunc = hiddenFunc;

        //        layers = new Layer[nodeCount];
        //        weights = new WeightMatrix[nodeCount];
        //        outBiases = new double[outDim];
        //        outWeights = new WeightMatrix[outDim];

        //        for (int i = 0; i < nodeCount; i++)
        //        {
        //            layers[i] = new Layer(hiddenDim);
        //            weights[i] = new DenseWeightMatrix(hiddenDim + inDim, hiddenDim);
        //        }

        //        for (int i = 0; i < outDim; i++)
        //            outWeights[i] = new DenseWeightMatrix(hiddenDim, 1);
        //    }

        //    public void WeightAssignForEach(Func<double, double> func)
        //    {
        //        for (int i = 0; i < weights.LongLength; i++)
        //            weights[i].AssignForEach((inIndex, outIndex, weight) => func(weight));

        //        for (int i = 0; i < outWeights.LongLength; i++)
        //            outWeights[i].AssignForEach((inIndex, outIndex, weight) => func(weight));
        //    }

        //    public void BiasAssignForEach(Func<double, double> func)
        //    {
        //        for (int i = 0; i < layers.LongLength; i++)
        //            for (int j = 0; j < layers[i].dim; j++)
        //                layers[i].SetBias(j, func(layers[i].GetBias(j)));

        //        for (int i = 0; i < outBiases.Length; i++)
        //            outBiases[i] = func(outBiases[i]);
        //    }

        //    public void GradientDescent(double[][] sampleOutputs, ForwardResult forwardLog, double learningRate)
        //    {
        //        if (!(forwardLog is RecurrentForwardResult))
        //            throw new Exception("hell nah");

        //        RecurrentForwardResult log = (RecurrentForwardResult)forwardLog;

        //        double[][] errors = new double[sampleOutputs.Length][];

        //        // Derivative of || 0.5 * (y - h(inputs))^2 ||
        //        for (int sample = 0; sample < sampleOutputs.Length; sample++)
        //        {
        //            errors[sample] = new double[sampleOutputs[0].Length];

        //            for (int i = 0; i < sampleOutputs[0].Length; i++)
        //            {
        //                errors[sample][i] = (log.outputs[sample][i] - sampleOutputs[sample][i]) * ActivationLayer.ActivationDifferential(outFunc, log.layerInputs[nodeCount - outDim + i][2][sample][0] + outBiases[i]);
        //            }
        //        }

        //        GradientDescentLayers(errors, new double[errors.Length][], log, learningRate, nodeCount - 1);
        //    }

        //    /// <summary>
        //    /// <i>For recursion purpose only</i>. Backpropagates and updates specified layers
        //    /// </summary>
        //    /// <param name="errors">Error vector of the commencing layer (or <b>fromLayer</b>) with [sample][outIndex]</param>
        //    /// <param name="fromlayer"><i>For recursion purpose only</i>. Going backwards from the given <b>fromLayer</b> index</param>
        //    private void GradientDescentLayers(double[][] errors, double[][] hiddenErrors, RecurrentForwardResult forwardLog, double learningRate, int fromNode)
        //    {
        //        if (fromNode < 0)
        //            return;
        //        else if (fromNode == nodeCount - 1)
        //            for (int i = 0; i < errors.Length; i++)
        //                hiddenErrors[i] = new double[hiddenDim];

        //        int outNodeIndex = fromNode - (nodeCount - outDim);

        //        if (outNodeIndex >= 0)
        //        {
        //            for (int i = 0; i < errors.Length; i++) {
        //                double outBiasTemp = outBiases[outNodeIndex];
        //                outBiases[outNodeIndex] -= errors[i][outNodeIndex] * learningRate;
        //                outWeights[outNodeIndex].AssignForEach(
        //                    (inIndex, outIndex, weight) =>
        //                    {
        //                        hiddenErrors[i][inIndex] += ActivationLayer.ActivationDifferential(outFunc, forwardLog.layerInputs[fromNode][2][i][0] + outBiasTemp) * weight;
        //                        hiddenErrors[i][inIndex] *= ActivationLayer.ActivationDifferential(hiddenFunc, forwardLog.layerInputs[fromNode][1][i][inIndex] + layers[fromNode].GetBias(inIndex));
        //                        layers[fromNode].SetBias(inIndex, 
        //                            layers[fromNode].GetBias(inIndex) - 
        //                            learningRate * 
        //                            hiddenErrors[i][inIndex]);

        //                        return (1 - weightDecay) * weight -
        //                        errors[i][outNodeIndex] *
        //                        ActivationLayer.ActivationDifferential(outFunc, forwardLog.layerInputs[fromNode][2][i][0] + outBiasTemp) *
        //                        layers[fromNode].ForwardComp(forwardLog.layerInputs[fromNode][1][i][inIndex] + layers[fromNode].GetBias(inIndex)) *
        //                        learningRate;
        //                    });
        //            }
        //        }

        //        for (int i = 0; i < errors.Length; i++)
        //        {
        //            double[] weightAccum = new double[hiddenDim];
        //            weights[fromNode].AssignForEach(
        //                (inIndex, outIndex, weight) =>
        //                {
        //                    if(inIndex < hiddenDim)
        //                        weightAccum[inIndex] += weight;

        //                    return (1 - weightDecay) * weight -
        //                    hiddenErrors[i][outIndex] *
        //                    forwardLog.layerInputs[fromNode][0][i][inIndex] *
        //                    learningRate;
        //                });
        //            for(int j = 0; j < hiddenDim; j++)
        //                hiddenErrors[i][j] *= weightAccum[j];
        //        }

        //        GradientDescentLayers(errors, hiddenErrors, forwardLog, learningRate, fromNode - 1);
        //    }

        //    // layerIndex : {
        //    //      0: inputs,
        //    //      1: hidden,
        //    //      2: outputs
        //    // }
        //    // layerInputs = [nodeIndex][layerIndex][sample][perceptron]
        //    // inputs = [sample][nodeIndex][inputs]
        //    public ForwardResult Forward(IForwardInput[] inputs)
        //    {
        //        double[][][][] log = new double[nodeCount][][][];
        //        double[][] result = new double[inputs.Length][];

        //        for (int i = 0; i < nodeCount; i++)
        //        {
        //            if (i < nodeCount - outDim)
        //                log[i] = new double[2][][];
        //            else
        //                log[i] = new double[3][][];

        //            for (int j = 0; j < log[i].Length; j++)
        //                log[i][j] = new double[inputs.Length][];
        //        }

        //        for(int i = 0; i < inputs.Length; i++)
        //            result[i] = Forward(((RecurrentForwardInput)inputs[i]).recurrent, ref log, i);

        //        return new RecurrentForwardResult(log, result);
        //    }

        //    // layerIndex : {
        //    //      0: inputs,
        //    //      1: hidden,
        //    //      2: outputs
        //    // }
        //    // layerInputs = [nodeIndex][layerIndex][sample = 1][perceptron]
        //    // inputs = [NodeIndex][inputs]
        //    public double[] Forward(double[][] inputs, ref double[][][][] layerInputs, int logSampleIndex = 0)
        //    {
        //        layerInputs[0][0][logSampleIndex] = new double[hiddenDim + inDim];
        //        for (int j = hiddenDim; j < hiddenDim + inDim; j++)
        //            layerInputs[0][0][logSampleIndex][j] = inputs[0][j - hiddenDim];

        //        layerInputs[0][1][logSampleIndex] = weights[0].Forward(layerInputs[0][0][logSampleIndex]);

        //        for (int i = 1; i < nodeCount; i++)
        //        {
        //            layerInputs[i][0][logSampleIndex] = new double[hiddenDim + inDim];

        //            for (int j = 0; j < hiddenDim; j++)
        //                layerInputs[i][0][logSampleIndex][j] = ActivationLayer.ForwardActivation(hiddenFunc, layers[i - 1].ForwardComp(layerInputs[i - 1][1][logSampleIndex][j] + layers[i - 1].GetBias(j)));
        //            for (int j = hiddenDim; j < hiddenDim + inDim; j++)
        //                layerInputs[i][0][logSampleIndex][j] = inputs[i][j - hiddenDim];

        //            layerInputs[i][1][logSampleIndex] = weights[i].Forward(layerInputs[i][0][logSampleIndex]);

        //            if (i >= nodeCount - outDim)
        //            {
        //                layerInputs[i][2][logSampleIndex] = new double[1];
        //                layerInputs[i][2][logSampleIndex] = outWeights[i - (nodeCount - outDim)].Forward(layers[i].Forward(layerInputs[i][1][logSampleIndex]));
        //            }
        //        }

        //        double[] result = new double[outDim];

        //        for (int i = 0; i < outDim; i++)
        //            result[i] = ActivationLayer.ForwardActivation(outFunc, layerInputs[i + (nodeCount - outDim)][2][0][0] + outBiases[i]);

        //        return result;
        //    }
        //}

        public class DenseNeuralNetwork : INeuralNetwork
        {
            public int inDim { get; private set; }
            public int outDim { get; private set; }
            public Optimizer optimizer { get; private set; }
            public Layer[] layers { get; private set; }
            public WeightMatrix[] weights { get; private set; }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, double learningRate, bool disposeAfterwards = true) : base()
            {
                Tuple<Layer[], WeightMatrix[]> bundle = builder.Build();

                this.layers = bundle.Item1;
                this.weights = bundle.Item2;
                this.optimizer = new SGD(learningRate);
                this.inDim = layers[0].dim;
                this.outDim = layers[layers.LongLength - 1].dim;

                foreach (var layer in layers)
                    layer.Build(this);

                foreach (var weight in weights)
                    weight.Build(this);

                optimizer.Init(this);

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, Optimizer optimizer, bool disposeAfterwards = true) : base()
            {
                Tuple<Layer[], WeightMatrix[]> bundle = builder.Build();

                this.layers = bundle.Item1;
                this.weights = bundle.Item2;
                this.optimizer = optimizer;
                this.inDim = layers[0].dim;
                this.outDim = layers[layers.LongLength - 1].dim;

                foreach (var layer in layers)
                    layer.Build(this);

                foreach (var weight in weights)
                    weight.Build(this);

                optimizer.Init(this);

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public void WeightAssignForEach(Func<double, double> func)
            {
                for (int i = 0; i < weights.LongLength; i++)
                    weights[i].AssignForEach((inIndex, outIndex, weight) => func(weight));
            }

            public void BiasAssignForEach(Func<double, double> func)
            {
                for (int i = 0; i < layers.LongLength; i++)
                    for (int j = 0; j < layers[i].dim; j++)
                        layers[i].SetBias(j, func(layers[i].GetBias(j)));
            }

            /// <summary>
            /// Backpropagates and updates weights, biases
            /// </summary>
            public void GradientDescent(double[][] sampleOutputs, ForwardResult forwardLog)
            {
                if (!(forwardLog is DenseForwardResult))
                    throw new Exception("hell nah");

                DenseForwardResult log = (DenseForwardResult)forwardLog;

                double[][] errors = new double[sampleOutputs.Length][];

                // Derivative of || 0.5 * (y - h(inputs))^2 ||
                for (int sample = 0; sample < sampleOutputs.Length; sample++)
                {
                    errors[sample] = new double[sampleOutputs[0].Length];

                    for (int i = 0; i < sampleOutputs[0].Length; i++)
                        errors[sample][i] = log.outputs[sample][i] - sampleOutputs[sample][i];
                }

                layers[layers.Length - 1].GradientDescent(ref errors, forwardLog, optimizer);

                for (int i = layers.Length - 2; i > -1; i--)
                {
                    weights[i].GradientDescent(ref errors, forwardLog, optimizer);
                    layers[i].GradientDescent(ref errors, forwardLog, optimizer);
                }
            }

            public DenseForwardResult Forward(double[] inputs)
            {
                double[][] wrapInputs = new double[1][];
                wrapInputs[0] = inputs;

                double[][][] layerInputs = new double[layers.LongLength][][];
                double[][] outputs = ForwardLayers(wrapInputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseForwardResult(layerInputs, outputs);
            }

            public ForwardResult Forward(IForwardInput[] inputs)
            {
                double[][][] layerInputs = new double[layers.LongLength][][];
                double[][] outputs = ForwardLayers(inputs, layers.Length - 1, 0, ref layerInputs);

                return new DenseForwardResult(layerInputs, outputs);
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

            double[][] ForwardLayers(IForwardInput[] inputs, int toLayer, int fromLayer, ref double[][][] layerInputs)
            {
                if (fromLayer < toLayer)
                    layerInputs[toLayer] = weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                {
                    double[][] result = new double[inputs.Length][];
                    for (int i = 0; i < inputs.Length; i++)
                        result[i] = ((DenseForwardInput)inputs[i]).dense;

                    layerInputs[toLayer] = result;
                }

                return layers[toLayer].Forward(layerInputs[toLayer]);
            }

        }

        public class DenseNeuralNetworkBuilder : INeuralNetworkBuilder
        {
            public List<Layer> layers;

            public DenseNeuralNetworkBuilder(int inputDim) 
            {
                layers = new List<Layer>();

                layers.Add(new Layer(inputDim, false));
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

            public Tuple<Layer[], WeightMatrix[]> Build()
            {
                WeightMatrix[] weights = new WeightMatrix[layers.Count - 1];

                for (int i = 1; i < layers.Count; i++)
                {
                    if (layers[i - 1] is ForwardLayer && ((ForwardLayer)layers[i - 1]).port != ForwardLayer.ForwardPort.In)
                        weights[i - 1] = layers[i - 1].GenerateWeightMatrix();
                    else
                        weights[i - 1] = layers[i].GenerateWeightMatrix();
                }

                return (layers.ToArray(), weights).ToTuple();
            }

            public void Dispose()
            {
                GC.SuppressFinalize(this);
            }
        }

        public interface INeuralNetworkBuilder : IDisposable
        {
            public abstract Tuple<Layer[], WeightMatrix[]> Build();
        }

        #region Optimizer

        public interface IBatchNormOptimizable
        {
            public Dictionary<int, int> bnIndexLookup { get; }

            public double GammaUpdate(int layerIndex, double gradient);

            public double BetaUpdate(int layerIndex, double gradient);
        }

        public abstract class Optimizer
        {
            public DenseNeuralNetwork network;
            public double weightDecay;

            public Optimizer(double weightDecay = 0)
            {
                this.weightDecay = weightDecay;
            }

            public virtual void Init(DenseNeuralNetwork network)
            {
                this.network = network;
            }

            public abstract double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient);

            public abstract double BiasUpdate(int layerIndex, int perceptron, double gradient);
        }

        public class SGD : Optimizer, IBatchNormOptimizable
        {
            public double learningRate;
            public Dictionary<int, int> bnIndexLookup { get; private set; }

            public SGD(double learningRate, double weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                return network.weights[weightsIndex].GetWeight(inIndex, outIndex) * (1 - weightDecay) - gradient * learningRate;
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                return network.layers[layerIndex].GetBias(perceptron) - gradient * learningRate;
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                return ((BatchNormLayer)network.layers[layerIndex]).gamma - gradient * learningRate;
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                return ((BatchNormLayer)network.layers[layerIndex]).beta - gradient * learningRate;
            }
        }

        public class Momentum : Optimizer, IBatchNormOptimizable
        {
            public double[][][] weightMomentum;
            public double[][] biasMomentum;
            public double learningRate, beta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public double[] gammaMomentum, betaMomentum;

            public Momentum(double beta = 0.9d, double learningRate = 0.01d, double weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta = beta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                weightMomentum = new double[network.weights.Length][][];
                for (int i = 0; i < network.weights.Length; i++)
                {
                    weightMomentum[i] = new double[network.weights[i].outDim][];
                    for (int j = 0; j < network.weights[i].outDim; j++)
                    {
                        weightMomentum[i][j] = new double[network.weights[i].inDim];
                        for (int k = 0; k < network.weights[i].inDim; k++)
                            weightMomentum[i][j][k] = 0.000001d; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                biasMomentum = new double[network.layers.Length][];
                for (int i = 0; i < network.layers.Length; i++)
                {
                    biasMomentum[i] = new double[network.layers[i].dim];
                    for (int j = 0; j < network.layers[i].dim; j++)
                        biasMomentum[i][j] = 0.000001d; // epsilon = 10^-6

                    if (network.layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                gammaMomentum = new double[bnIndexLookup.Count];
                betaMomentum = new double[bnIndexLookup.Count];
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                weightMomentum[weightsIndex][outIndex][inIndex] = 
                    beta * weightMomentum[weightsIndex][outIndex][inIndex] + 
                    (1 - beta) * (gradient + weightDecay * network.weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.weights[weightsIndex].GetWeight(inIndex, outIndex) - learningRate * weightMomentum[weightsIndex][outIndex][inIndex];
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                biasMomentum[layerIndex][perceptron] = beta * biasMomentum[layerIndex][perceptron] + (1 - beta) * gradient;

                return network.layers[layerIndex].GetBias(perceptron) - learningRate * biasMomentum[layerIndex][perceptron];
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                gammaMomentum[bnIndexLookup[layerIndex]] = beta * gammaMomentum[bnIndexLookup[layerIndex]] + (1 - beta) * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).gamma - learningRate * gammaMomentum[bnIndexLookup[layerIndex]];
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                betaMomentum[bnIndexLookup[layerIndex]] = beta * betaMomentum[bnIndexLookup[layerIndex]] + (1 - beta) * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).beta - learningRate * betaMomentum[bnIndexLookup[layerIndex]];
            }
        } 
        
        public class RMSprop : Optimizer, IBatchNormOptimizable
        {
            public double[][][] accumWeightGrad;
            public double[][] accumBiasGrad;
            public double learningRate, beta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public double[] accumGammaGrad, accumBetaGrad;

            public RMSprop(double beta = 0.99d, double learningRate = 0.01d, double weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta = beta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new double[network.weights.Length][][];
                for (int i = 0; i < network.weights.Length; i++)
                {
                    accumWeightGrad[i] = new double[network.weights[i].outDim][];
                    for (int j = 0; j < network.weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new double[network.weights[i].inDim];
                        for (int k = 0; k < network.weights[i].inDim; k++)
                            accumWeightGrad[i][j][k] = 0.000001d; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new double[network.layers.Length][];
                for (int i = 0; i < network.layers.Length; i++)
                {
                    accumBiasGrad[i] = new double[network.layers[i].dim];
                    for (int j = 0; j < network.layers[i].dim; j++)
                        accumBiasGrad[i][j] = 0.000001d; // epsilon = 10^-6

                    if (network.layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                accumGammaGrad = new double[bnIndexLookup.Count];
                accumBetaGrad = new double[bnIndexLookup.Count];
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] =
                    beta * accumWeightGrad[weightsIndex][outIndex][inIndex] +
                    (1 - beta) * (gradient * gradient + weightDecay * network.weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.weights[weightsIndex].GetWeight(inIndex, outIndex) - (learningRate * gradient / Math.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex]));
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                accumBiasGrad[layerIndex][perceptron] = beta * accumBiasGrad[layerIndex][perceptron] + (1 - beta) * gradient * gradient;

                return network.layers[layerIndex].GetBias(perceptron) - (learningRate * gradient / Math.Sqrt(accumBiasGrad[layerIndex][perceptron]));
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] = beta * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - beta) * gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).gamma - (learningRate * gradient / Math.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]]));
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] = beta * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - beta) * gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).beta - (learningRate * gradient / Math.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]]));
            }
        }
        
        public class Adam : Optimizer, IBatchNormOptimizable
        {
            public double[][][] accumWeightGrad, weightMomentum;
            public double[][] accumBiasGrad, biasMomentum;
            public double learningRate, beta1, beta2;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public double[] accumGammaGrad, gammaMomentum, accumBetaGrad, betaMomentum;

            public Adam(double beta1 = 0.9d, double beta2 = 0.99d, double learningRate = 0.01d, double weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta1 = beta1;
                this.beta2 = beta2;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                weightMomentum = new double[network.weights.Length][][];
                accumWeightGrad = new double[network.weights.Length][][];
                for (int i = 0; i < network.weights.Length; i++)
                {
                    weightMomentum[i] = new double[network.weights[i].outDim][];
                    accumWeightGrad[i] = new double[network.weights[i].outDim][];
                    for (int j = 0; j < network.weights[i].outDim; j++)
                    {
                        weightMomentum[i][j] = new double[network.weights[i].inDim];
                        accumWeightGrad[i][j] = new double[network.weights[i].inDim];
                        for (int k = 0; k < network.weights[i].inDim; k++)
                        {
                            weightMomentum[i][j][k] = 0.000001d; // epsilon = 10^-6
                            accumWeightGrad[i][j][k] = 0.000001d; // epsilon = 10^-6
                        }
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                biasMomentum = new double[network.layers.Length][];
                accumBiasGrad = new double[network.layers.Length][];
                for (int i = 0; i < network.layers.Length; i++)
                {
                    biasMomentum[i] = new double[network.layers[i].dim];
                    accumBiasGrad[i] = new double[network.layers[i].dim];
                    for (int j = 0; j < network.layers[i].dim; j++)
                    {
                        biasMomentum[i][j] = 0.000001d; // epsilon = 10^-6
                        accumBiasGrad[i][j] = 0.000001d; // epsilon = 10^-6
                    }

                    if (network.layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                gammaMomentum = new double[bnIndexLookup.Count];
                accumGammaGrad = new double[bnIndexLookup.Count];
                betaMomentum = new double[bnIndexLookup.Count];
                accumBetaGrad = new double[bnIndexLookup.Count];
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                weightMomentum[weightsIndex][outIndex][inIndex] =
                    beta1 * weightMomentum[weightsIndex][outIndex][inIndex] +
                    (1 - beta1) * (gradient + weightDecay * network.weights[weightsIndex].GetWeight(inIndex, outIndex));

                accumWeightGrad[weightsIndex][outIndex][inIndex] =
                    beta2 * accumWeightGrad[weightsIndex][outIndex][inIndex] +
                    (1 - beta2) * (gradient * gradient + weightDecay * network.weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.weights[weightsIndex].GetWeight(inIndex, outIndex) - (learningRate * weightMomentum[weightsIndex][outIndex][inIndex] / Math.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex]));
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                biasMomentum[layerIndex][perceptron] = beta1 * biasMomentum[layerIndex][perceptron] + (1 - beta1) * gradient;
                accumBiasGrad[layerIndex][perceptron] = beta2 * accumBiasGrad[layerIndex][perceptron] + (1 - beta2) * gradient * gradient;

                return network.layers[layerIndex].GetBias(perceptron) - (learningRate * biasMomentum[layerIndex][perceptron] / Math.Sqrt(accumBiasGrad[layerIndex][perceptron]));
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                gammaMomentum[bnIndexLookup[layerIndex]] = beta1 * gammaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumGammaGrad[bnIndexLookup[layerIndex]] = beta2 * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).gamma - (learningRate * gammaMomentum[bnIndexLookup[layerIndex]] / Math.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]]));
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                betaMomentum[bnIndexLookup[layerIndex]] = beta1 * betaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumBetaGrad[bnIndexLookup[layerIndex]] = beta2 * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).beta - (learningRate * betaMomentum[bnIndexLookup[layerIndex]] / Math.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]]));
            }
        }

        public class AdaGrad : Optimizer, IBatchNormOptimizable
        {
            public double[][][] accumWeightGrad;
            public double[][] accumBiasGrad;
            public double eta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public double[] accumGammaGrad, accumBetaGrad;

            public AdaGrad(double eta = 0.01d, double weightDecay = 0) : base(weightDecay)
            {
                this.eta = eta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new double[network.weights.Length][][];
                for (int i = 0; i < network.weights.Length; i++)
                {
                    accumWeightGrad[i] = new double[network.weights[i].outDim][];
                    for (int j = 0; j < network.weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new double[network.weights[i].inDim];
                        for (int k = 0; k < network.weights[i].inDim; k++)
                            accumWeightGrad[i][j][k] = 0.000001d; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new double[network.layers.Length][];
                for (int i = 0; i < network.layers.Length; i++)
                {
                    accumBiasGrad[i] = new double[network.layers[i].dim];
                    for (int j = 0; j < network.layers[i].dim; j++)
                        accumBiasGrad[i][j] = 0.000001d; // epsilon = 10^-6

                    if (network.layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                accumGammaGrad = new double[bnIndexLookup.Count];
                accumBetaGrad = new double[bnIndexLookup.Count];
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] += gradient * gradient;

                return (1 - weightDecay) * network.weights[weightsIndex].GetWeight(inIndex, outIndex) - (eta / Math.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex])) * gradient;
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                accumBiasGrad[layerIndex][perceptron] += gradient * gradient;

                return network.layers[layerIndex].GetBias(perceptron) - (eta / Math.Sqrt(accumBiasGrad[layerIndex][perceptron])) * gradient;
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] += gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).gamma - (eta / Math.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]])) * gradient;
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] += gradient * gradient;

                return ((BatchNormLayer)network.layers[layerIndex]).beta - (eta / Math.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]])) * gradient;
            }
        }

        public class AdaDelta : Optimizer, IBatchNormOptimizable
        {
            public double[][][] accumWeightGrad, accumRescaledWeightGrad;
            public double[][] accumBiasGrad, accumRescaledBiasGrad;
            public double rho;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public double[] accumGammaGrad, accumBetaGrad, accumRescaledGammaGrad, accumRescaledBetaGrad;

            public AdaDelta(double rho = 0.9d, double weightDecay = 0) : base(weightDecay)
            {
                this.rho = rho;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new double[network.weights.Length][][];
                accumRescaledWeightGrad = new double[network.weights.Length][][];
                for (int i = 0; i < network.weights.Length; i++)
                {
                    accumWeightGrad[i] = new double[network.weights[i].outDim][];
                    accumRescaledWeightGrad[i] = new double[network.weights[i].outDim][];
                    for (int j = 0; j < network.weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new double[network.weights[i].inDim];
                        accumRescaledWeightGrad[i][j] = new double[network.weights[i].inDim];
                        for (int k = 0; k < network.weights[i].inDim; k++)
                        {
                            accumWeightGrad[i][j][k] = 0.000001d; // epsilon = 10^-6
                            accumRescaledWeightGrad[i][j][k] = 0.0000001d;
                        }
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new double[network.layers.Length][];
                accumRescaledBiasGrad = new double[network.layers.Length][];
                for (int i = 0; i < network.layers.Length; i++)
                {
                    accumBiasGrad[i] = new double[network.layers[i].dim];
                    accumRescaledBiasGrad[i] = new double[network.layers[i].dim];
                    for (int j = 0; j < network.layers[i].dim; j++)
                    {
                        accumBiasGrad[i][j] = 0.000001d; // epsilon = 10^-6
                        accumRescaledBiasGrad[i][j] = 0.0000001d;
                    }

                    if (network.layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                bool hasBatchNormLayer = false;
                foreach (Layer layer in network.layers)
                    if (layer is BatchNormLayer)
                    {
                        hasBatchNormLayer = true;
                        break;
                    }

                if (!hasBatchNormLayer)
                    return;

                accumGammaGrad = new double[bnIndexLookup.Count];
                accumBetaGrad = new double[bnIndexLookup.Count];
                accumRescaledGammaGrad = new double[bnIndexLookup.Count];
                accumRescaledBetaGrad = new double[bnIndexLookup.Count];
            }

            public override double WeightUpdate(int weightsIndex, int inIndex, int outIndex, double gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] = rho * accumWeightGrad[weightsIndex][outIndex][inIndex] + (1 - rho) * gradient * gradient;

                double rescaledGrad = 
                    Math.Sqrt(accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] / accumWeightGrad[weightsIndex][outIndex][inIndex]) * gradient + 
                    weightDecay * network.weights[weightsIndex].GetWeight(inIndex, outIndex);
                accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] = rho * accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] + (1 - rho) * rescaledGrad * rescaledGrad;

                return network.weights[weightsIndex].GetWeight(inIndex, outIndex) - rescaledGrad;
            }

            public override double BiasUpdate(int layerIndex, int perceptron, double gradient)
            {
                accumBiasGrad[layerIndex][perceptron] = rho * accumBiasGrad[layerIndex][perceptron] + (1 - rho) * gradient * gradient;

                double rescaledGrad = Math.Sqrt(accumRescaledBiasGrad[layerIndex][perceptron] / accumBiasGrad[layerIndex][perceptron]) * gradient;
                accumRescaledBiasGrad[layerIndex][perceptron] = rho * accumRescaledBiasGrad[layerIndex][perceptron] + (1 - rho) * rescaledGrad * rescaledGrad;

                return network.layers[layerIndex].GetBias(perceptron) - rescaledGrad;
            }

            public double GammaUpdate(int layerIndex, double gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] = rho * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * gradient * gradient;

                double rescaledGrad = Math.Sqrt(accumRescaledGammaGrad[bnIndexLookup[layerIndex]] / accumGammaGrad[bnIndexLookup[layerIndex]]) * gradient;
                accumRescaledGammaGrad[bnIndexLookup[layerIndex]] = rho * accumRescaledGammaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * rescaledGrad * rescaledGrad;

                return ((BatchNormLayer)network.layers[layerIndex]).gamma - rescaledGrad;
            }

            public double BetaUpdate(int layerIndex, double gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] = rho * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * gradient * gradient;

                double rescaledGrad = Math.Sqrt(accumRescaledBetaGrad[bnIndexLookup[layerIndex]] / accumBetaGrad[bnIndexLookup[layerIndex]]) * gradient;
                accumRescaledBetaGrad[bnIndexLookup[layerIndex]] = rho * accumRescaledBetaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * rescaledGrad * rescaledGrad;

                return ((BatchNormLayer)network.layers[layerIndex]).beta - rescaledGrad;
            }
        }

        #endregion

        #region Layer

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

            public BatchNormLayer(ForwardPort port) : base(ActivationFunc.Custom, port, false) { }

            public override double[][] Forward(double[][] inputs)
            {
                int sampleSize = inputs.Length;
                double[][] result = new double[sampleSize][];

                for (int sample = 0; sample < result.Length; sample++)
                    result[sample] = new double[dim];

                for (int i = 0; i < dim; i++)
                {
                    double mean = 0, variance = 0;

                    for (int sample = 0; sample < sampleSize; sample++)
                        mean += inputs[sample][i];
                    mean /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variance += Math.Pow(inputs[sample][i] - mean, 2);
                    variance /= sampleSize;

                    for (int sample = 0; sample < result.Length; sample++)
                        result[sample][i] = ForwardComp(Standardize(inputs[sample][i], mean, variance)); 
                }

                return result;
            }

            public override double ForwardComp(double x)
            {
                return base.ForwardComp(x * gamma + beta);
            }

            public override void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer)
            {
                if (layerIndex == 0)
                    return;

                Layer prevLayer = network.layers[layerIndex - 1];
                DenseForwardResult log = (DenseForwardResult)result;

                int sampleSize = errors.Length;

                double[] means = new double[dim];
                double[] variances = new double[dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++)
                        means[i] += log.layerInputs[layerIndex][sample][i];
                    means[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variances[i] += Math.Pow(log.layerInputs[layerIndex][sample][i] - means[i], 2);
                    variances[i] /= sampleSize;
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
                        dgamma[i] += errors[sample][i] * Standardize(log.layerInputs[layerIndex][sample][i], means[i], variances[i]);

                        dvariances[i] += errors[sample][i] * (log.layerInputs[layerIndex][sample][i] - means[i]);
                        dmeans[i] += errors[sample][i];
                    }

                    dvariances[i] *= (-0.5d) * gamma * Math.Pow(variances[i], -1.5d);
                    dvariances[i] += 0.000001d;

                    dmeans[i] *= (gamma * sampleSize) / (Math.Sqrt(variances[i]) * dvariances[i] * 2);
                    // dmeans[i] = (-gamma) / Math.Sqrt(variances[i]); 
                    // dmeans[i] /= dvariances[i] * (-2) * (1 / sampleSize); 

                    for (int sample = 0; sample < sampleSize; sample++)
                        dmeans[i] += log.layerInputs[layerIndex][sample][i] - means[i];
                    dmeans[i] *= dvariances[i] * (-2);
                    dmeans[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        errors[sample][i] =
                            (errors[sample][i] * gamma) / Math.Sqrt(variances[i]) +
                            dmeans[i] / sampleSize +
                            (2 * dvariances[i] * (log.layerInputs[layerIndex][sample][i] - means[i])) / sampleSize;
                }

                for (int i = 0; i < dim; i++)
                {
                    gamma = ((IBatchNormOptimizable)optimizer).GammaUpdate(layerIndex, dgamma[i]);
                    beta = ((IBatchNormOptimizable)optimizer).BetaUpdate(layerIndex, dbeta[i]);
                }
            }

            public static double Standardize(double x, double mean, double variance, double zeroSub = 0.000001) => variance != 0 ? (x - mean) / Math.Sqrt(variance) : (x - mean) / zeroSub;

        }

        public class NormalizationLayer : ForwardLayer
        {
            public double gamma, beta;

            public NormalizationLayer(double min, double max, ForwardPort port) : base(ActivationFunc.Custom, port, false)
            {
                gamma = 1 / (max - min);
                beta = -min;
            }

            public override void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer) { }

            public override double ForwardComp(double x)
            {
                return base.ForwardComp(x * gamma + beta);
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

            public ForwardLayer(ActivationFunc func, ForwardPort port, bool useBias = true) : base(-1, func, useBias) 
            {
                this.port = port;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                switch (port) 
                {
                    case ForwardPort.In:
                        dim = network.layers[layerIndex - 1].dim;
                        break; 
                    case ForwardPort.Out:
                        dim = network.layers[layerIndex + 1].dim;
                        break;
                    case ForwardPort.Both:
                        if (network.layers[layerIndex - 1].dim != network.layers[layerIndex + 1].dim)
                            throw new Exception("Nah forward layer dim");
                        dim = network.layers[layerIndex + 1].dim;
                        break;
                }

                biases = new double[dim];
            }

            public override WeightMatrix GenerateWeightMatrix()
            {
                return new ForwardWeightMatrix(useBias);
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

            public static double ActivationDifferential(ActivationFunc func, double x)
            {
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        double sigmoid = ForwardActivation(func, x);
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

            public static double ForwardActivation(ActivationFunc func, double x)
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
        }

        public class Layer 
        {
            public readonly bool useBias;

            public int dim { get; protected set; } = -1;
            public INeuralNetwork network { get; protected set; }

            protected int layerIndex = -1;
            protected double[] biases;

            public Layer(int dim, bool useBias = true)
            {
                this.dim = dim;
                this.useBias = useBias;

                if (dim != -1)
                    biases = new double[dim];
            }
            
            public Layer(double[] biases)
            {
                this.dim = biases.Length;
                this.biases = biases;
            }

            public virtual void Build(INeuralNetwork network)
            {
                this.network = network;
                for (int i = 0; i < network.layers.Length; i++)
                    if (network.layers[i] == this)
                    {
                        layerIndex = i;
                        return;
                    }

                throw new Exception("nah layer findings");
            }

            public virtual double GetBias(int index) => useBias ? biases[index] : 0;

            public virtual void SetBias(int index, double value) => biases[index] = useBias ? value : 0; 

            /// <returns>Returns descended errors</returns>
            public virtual void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer)
            {
                if (!useBias)
                    return;

                DenseForwardResult log = ((DenseForwardResult)result);

                for (int i = 0; i < dim; i++)
                {
                    double gradSum = 0;

                    // bias update
                    for (int sample = 0; sample < errors.Length; sample++)
                    {
                        errors[sample][i] *= FunctionDifferential(log.layerInputs[layerIndex][sample][i] + GetBias(i));
                        gradSum += errors[sample][i];
                    }

                    SetBias(i, optimizer.BiasUpdate(layerIndex, i, gradSum));
                }
            }

            public virtual double[][] Forward(double[][] inputs)
            {
                double[][] result = new double[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new double[dim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < dim; j++)
                        result[i][j] = ForwardComp(inputs[i][j] + GetBias(j));

                return result;
            }

            public virtual double[] Forward(double[] inputs)
            {
                double[] result = new double[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = ForwardComp(inputs[i] + GetBias(i));

                return result;
            }

            public virtual double ForwardComp(double x) => x;

            /// <summary>
            /// Get <b>df(bias, x) / dx</b> such <b>x</b> can be another function
            /// </summary>
            public virtual double FunctionDifferential(double x) => 1;

            public virtual WeightMatrix GenerateWeightMatrix()
            {
                return new DenseWeightMatrix();
            }

            public static implicit operator Layer(int dim) => new Layer(dim);
        }

        #endregion

        #region Weight matrix

        public abstract class WeightMatrix
        {
            public int inDim { get; protected set; }
            public int outDim { get; protected set; }
            public INeuralNetwork network { get; protected set; }

            protected int weightsIndex;

            public virtual void Build(INeuralNetwork network)
            {
                this.network = network;
                for (int i = 0; i < network.weights.Length; i++)
                    if (network.weights[i] == this)
                    {
                        weightsIndex = i;
                        return;
                    }

                throw new Exception("nah weights findings");
            }

            public abstract void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer);

            public abstract double[] Forward(double[] inputs);

            public abstract double[][] Forward(double[][] inputs);

            public abstract double ForwardComp(double[] inputs, int outputIndex);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="value"><inIndex, outIndex, weightValue, returnValue></param>
            public abstract void AssignForEach(Func<int, int, double, double> value);

            public abstract bool TrySetWeight(int inIndex, int outIndex, double value);

            public abstract bool TryGetWeight(int inIndex, int outIndex, out double weight);

            public abstract double GetWeight(int inIndex, int outIndex);
        }

        public class ForwardWeightMatrix : WeightMatrix
        {
            public readonly bool useWeights;

            public double[] matrix;

            public int dim => inDim;

            public ForwardWeightMatrix(bool useWeights = true)
            {
                this.useWeights = useWeights;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                if (network.layers[weightsIndex] is ForwardLayer && ((ForwardLayer)network.layers[weightsIndex]).port != ForwardLayer.ForwardPort.In)
                    inDim = outDim = network.layers[weightsIndex].dim;
                else if (network.layers[weightsIndex + 1] is ForwardLayer && ((ForwardLayer)network.layers[weightsIndex + 1]).port != ForwardLayer.ForwardPort.Out)
                    inDim = outDim = network.layers[weightsIndex + 1].dim;
                else
                    throw new Exception("Nah forward weight dim");

                matrix = new double[dim];
            }

            public override void AssignForEach(Func<int, int, double, double> value)
            {
                for (int i = 0; i < dim; i++) {
                    if (useWeights)
                        matrix[i] = value(i, i, matrix[i]);
                    else
                        value(i, i, 1);
                }
            }

            public override void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer)
            {
                if (!useWeights) return;

                DenseForwardResult log = (DenseForwardResult)result;

                for (int i = 0; i < matrix.Length; i++)
                {
                    double weightErrorSum = 0;

                    for (int sample = 0; sample < errors.Length; sample++)
                    {
                        weightErrorSum += errors[sample][i] * network.layers[weightsIndex].ForwardComp(log.layerInputs[weightsIndex][sample][i] + network.layers[weightsIndex].GetBias(i));
                        errors[sample][i] *= matrix[i] * network.layers[weightsIndex].FunctionDifferential(log.layerInputs[weightsIndex][sample][i] + network.layers[weightsIndex].GetBias(i));
                    }

                    matrix[i] = optimizer.WeightUpdate(weightsIndex, i, i, weightErrorSum);
                }
            }

            public override double[] Forward(double[] inputs)
            {
                double[] result = new double[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = ForwardComp(inputs, i);

                return result;
            }
            
            public override double[][] Forward(double[][] inputs)
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

            public override double ForwardComp(double[] inputs, int outputIndex)
            {
                if (useWeights)
                    return inputs[outputIndex] * matrix[outputIndex];
                else
                    return inputs[outputIndex];
            }

            public override double GetWeight(int inIndex, int outIndex)
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

            public override bool TryGetWeight(int inIndex, int outIndex, out double weight)
            {
                if (useWeights)
                    weight = matrix[inIndex];
                else if (inIndex == outIndex)
                    weight = 1;
                else
                    weight = 0;

                return inIndex == outIndex && inIndex < dim;
            }

            public override bool TrySetWeight(int inIndex, int outIndex, double value)
            {
                if (useWeights && inIndex == outIndex && inIndex < dim)
                {
                    matrix[inIndex] = value;
                    return true;
                }

                return false;
            }
        }

        public class DenseWeightMatrix : WeightMatrix
        {
            public double[,] matrix;

            public DenseWeightMatrix() { }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                inDim = network.layers[weightsIndex].dim;
                outDim = network.layers[weightsIndex + 1].dim;

                matrix = new double[outDim, inDim];
            }

            public override void GradientDescent(ref double[][] errors, ForwardResult result, Optimizer optimizer)
            {
                DenseForwardResult log = (DenseForwardResult)result;
                Layer prevLayer = network.layers[weightsIndex];

                double[][] weightErrors = new double[errors.Length][];
                for (int i = 0; i < errors.Length; i++)
                    weightErrors[i] = new double[inDim];

                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                    {
                        double weightErrorSum = 0;

                        for (int sample = 0; sample < errors.Length; sample++)
                        {
                            weightErrorSum += errors[sample][i] * prevLayer.ForwardComp(log.layerInputs[weightsIndex][sample][j] + prevLayer.GetBias(j));
                            weightErrors[sample][j] += errors[sample][i] * matrix[i,j] * prevLayer.FunctionDifferential(log.layerInputs[weightsIndex][sample][j] + prevLayer.GetBias(j));
                        }

                        matrix[i,j] = optimizer.WeightUpdate(weightsIndex, j, i, weightErrorSum);
                    }

                errors = weightErrors;
            }

            public override bool TryGetWeight(int inIndex, int outIndex, out double weight)
            {
                weight = matrix[outIndex, inIndex];
                return true;
            }

            public override double GetWeight(int inIndex, int outIndex) => matrix[outIndex, inIndex];

            public override void AssignForEach(Func<int, int, double, double> value)
            {
                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        matrix[i, j] = value(j, i, matrix[i, j]);
            }

            public override double[] Forward(double[] inputs)
            {
                double[] result = new double[outDim];

                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        result[i] += inputs[j] * matrix[i, j];

                return result;
            }

            public override double[][] Forward(double[][] inputs)
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

            public override double ForwardComp(double[] inputs, int outputIndex)
            {
                double output = 0;

                for (int i = 0; i < inputs.LongLength; i++)
                    output += matrix[outputIndex, i] * inputs[i];

                return output;
            }

            public override bool TrySetWeight(int inIndex, int outIndex, double value)
            {
                matrix[outIndex, inIndex] = value; 
                return true;
            }
        }

        #endregion
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
                Dictionary<string, AdditionNumericDataInfo> numericInfos;

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

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveNumberDataFromCSV(string path, int retrieveAmount, out Dictionary<string, AdditionNumericDataInfo> numericInfos, params string[] retrieveCats)
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

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                Dictionary<string, AdditionNumericDataInfo> numericInfos;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                string[] cats;
                return RetrieveUDataFromCSV(path, info, out cats, out distinctEncodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, double>[] RetrieveUDataFromCSV(string path, UDataInfo info, out string[] categories, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                List<Dictionary<string, double>> data = new List<Dictionary<string, double>>();
                List<double[]> rawData = new List<double[]>();
                numericInfos = new Dictionary<string, AdditionNumericDataInfo>();

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
                            numericInfos.Add(categories[i], new AdditionNumericDataInfo(double.NaN, double.NaN, 0));
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

                        bool empty = false;
                        foreach (int index in iteratingIndices)
                            if (string.IsNullOrEmpty(rawDataLine[index]))
                            {
                                empty = true;
                                break;
                            }
                        if (empty) continue;

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

                                    double min = numericInfos[categories[iteratingIndices[i]]].min;
                                    double max = numericInfos[categories[iteratingIndices[i]]].max;

                                    if (min > dataLine[i] || double.IsNaN(min)) min = dataLine[i];
                                    if (max < dataLine[i] || double.IsNaN(max)) max = dataLine[i];

                                    numericInfos[categories[iteratingIndices[i]]] = new AdditionNumericDataInfo(min, max, numericInfos[categories[iteratingIndices[i]]].mean + dataLine[i]);

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

                    foreach (int i in iteratingIndices)
                        if (info.types[i] == DataType.Double)
                            numericInfos[categories[i]] = new AdditionNumericDataInfo(numericInfos[categories[i]].min, numericInfos[categories[i]].max, numericInfos[categories[i]].mean / rawData.Count);

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

        public enum NormalizationMode
        {
            MinMaxRange,
            DivideMean,
            CutMinDivideMean
        }

        public struct AdditionNumericDataInfo
        {
            public double min, max, mean;

            public AdditionNumericDataInfo(double min, double max, double mean)
            {
                this.min = min;
                this.max = max;
                this.mean = mean;
            }

            public double Normalize(NormalizationMode mode, double value)
            {
                switch (mode)
                {
                    case NormalizationMode.MinMaxRange:
                        return (value - min) / (max - min);
                    case NormalizationMode.DivideMean:
                        return value / mean;
                    case NormalizationMode.CutMinDivideMean: 
                        return (value - min) / (mean - min);
                    default:
                        return value;
                }
            }

            public double Denormalize(NormalizationMode mode, double value)
            {
                switch (mode)
                {
                    case NormalizationMode.MinMaxRange:
                        return value * (max - min) + min;
                    case NormalizationMode.DivideMean:
                        return value * mean;
                    case NormalizationMode.CutMinDivideMean: 
                        return (value - 1) * (mean - min) + min;
                    default:
                        return value;
                }
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

    namespace Utility
    {
        namespace Linear
        {
            public static class LinearMethod
            {
                public static Vector CGMethod(ISquareMatrix A, Vector b, double epsilon = 1E-6)
                {
                    if (A.dim != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");

                    Vector
                        result = new double[A.dim],
                        residual = Vector.Clone(b.content), preResidual = Vector.Clone(b.content),
                        direction = Vector.Clone(residual.content);

                    if (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * A, direction);

                        result += alpha * direction;
                        residual -= alpha * A * direction;
                    }

                    while (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double beta = Vector.Dot(residual, residual) / Vector.Dot(preResidual, preResidual);

                        direction = residual + beta * direction;

                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * A, direction);

                        preResidual.SetTo(residual);
                        result += alpha * direction;
                        residual -= alpha * A * direction;
                    }

                    return result.content;
                }

                public static Vector CGNEMethod(IMatrix A, Vector b, double epsilon = 1E-6)
                {
                    if (A.rowCount != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");
                    IMatrix At = A.Transpose;

                    return CGMethod((At * A).ToSquare, At * b, epsilon);
                }

                /// <returns>A tuple of a lower matrix and an upper LU factorizations respectively</returns>
                public static (TriangularMatrix, TriangularMatrix) IncompleteLUFac(ISquareMatrix A, double epsilon = 1E-3)
                {
                    TriangularMatrix lower = new TriangularMatrix(A.dim, false);
                    TriangularMatrix upper = new TriangularMatrix(A.dim, true);

                    for (int i = 0; i < A.dim; i++)
                        for (int j = 0; j <= i; j++)
                        {
                            if (A.Get(j, i) > epsilon)
                            {
                                // Row iterate
                                // j : row index
                                double rowSum = 0;
                                for (int k = 0; k < j; k++)
                                    rowSum += lower.Get(j, k) * upper.Get(k, i);
                                upper.Set(j, i, A.Get(j, i) - rowSum);
                            }

                            if (A.Get(i, j) > epsilon)
                            {
                                // Column iterate
                                // j : column index
                                if (i == j)
                                    lower.Set(i, j, 1);
                                else
                                {
                                    double colSum = 0;
                                    for (int k = 0; k < j; k++)
                                        colSum += lower.Get(i, k) * upper.Get(k, j);

                                    lower.Set(i, j, (A.Get(i, j) - colSum) / upper.Get(j, j));
                                }
                            }

                        }

                    return (lower, upper);
                }

                /// <returns>Lower triangular factorization</returns>
                public static TriangularMatrix IncompleteCholeskyFac(ISquareMatrix A, double epsilon = 1E-3)
                {
                    TriangularMatrix result = new TriangularMatrix(A.dim, false);

                    for (int row = 0; row < A.dim; row++)
                        for (int col = 0; col < row + 1; col++)
                        {
                            if (A.Get(row, col) < epsilon)
                            {
                                result.Set(row, col, 0);
                                continue;
                            }

                            double sum = 0;
                            for (int i = 0; i < col; i++)
                                sum += result.Get(row, i) * result.Get(col, i);

                            if (col == row)
                                result.Set(row, col, Math.Sqrt(A.Get(row, col) - sum));
                            else
                                result.Set(row, col, (A.Get(row, col) - sum) / result.Get(col, col));
                        }

                    return result;
                }
            } 

            public interface IMatrix
            {
                public int rowCount { get; }
                public int colCount { get; }

                public IMatrix Transpose { get; }
                public ISquareMatrix ToSquare { get; }

                public IMatrix Instance();
                public IMatrix InstanceT();

                public double Get(int row, int col);
                public bool Set(int row, int col, double value);

                public bool SetTo(IMatrix matrix);

                public IMatrix Clone();

                public static int NegOneRaiseTo(int num)
                {
                    return num % 2 == 0 ? 1 : -1;
                }

                public static IMatrix Identity(int dim)
                {
                    IMatrix identity = new DiagonalMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        identity.Set(i, i, 1);

                    return identity;
                }

                public static IMatrix Diag(params double[] nums)
                {
                    IMatrix matrix = new DiagonalMatrix(nums.Length);

                    for (int i = 0; i < nums.Length; i++)
                        matrix.Set(i, i, nums[i]);

                    return matrix;
                }

                public Vector Multiply(Vector vector);
                public Vector LeftMultiply(Vector vector);

                public IMatrix Multiply(IMatrix matrix);

                public IMatrix Add(double value);

                public IMatrix Subtract(double value);
                public IMatrix LeftSubtract(double value);

                public IMatrix Multiply(double value);

                public IMatrix Divide(double value);

                public static IMatrix operator *(IMatrix A, IMatrix B) => A.Multiply(B);
                public static Vector operator *(IMatrix matrix, Vector vector) => matrix.Multiply(vector);
                public static Vector operator *(Vector vector, IMatrix matrix) => matrix.LeftMultiply(vector);

                public static IMatrix operator +(IMatrix matrix, double value) => matrix.Add(value);
                public static IMatrix operator -(IMatrix matrix, double value) => matrix.Subtract(value);
                public static IMatrix operator *(IMatrix matrix, double value) => matrix.Multiply(value);
                public static IMatrix operator /(IMatrix matrix, double value) => matrix.Divide(value);

                public static IMatrix operator +(double value, IMatrix matrix) => matrix.Add(value);
                public static IMatrix operator -(double value, IMatrix matrix) => matrix.LeftSubtract(value);
                public static IMatrix operator *(double value, IMatrix matrix) => matrix.Multiply(value);
            }

            public interface ISquareMatrix : IMatrix
            {
                public int dim { get; }

                public ISquareMatrix Invert();

                public double Determinant();

                public double Cofactor(int row, int col);

                public ISquareMatrix Adjugate();

                public static double Cofactor(ISquareMatrix matrix, AdjugateSum adj, double multiplier, bool coefMultiply = false)
                {
                    if (multiplier == 0)
                        return 0;

                    if (matrix.dim - adj.Count == 2)
                    {
                        int sum = (matrix.dim - 1) * matrix.dim / 2;
                        int row1 = adj.smallestAdjugatableRow, row2 = sum - adj.rowSum - row1,
                            col1 = adj.smallestAdjugatableCol, col2 = sum - adj.colSum - col1;

                        return multiplier * (matrix.Get(row1, col1) * matrix.Get(row2, col2) - matrix.Get(row1, col2) * matrix.Get(row2, col1));
                    }

                    double result = 0;

                    for (int i = 0; i < matrix.dim; i++)
                    {
                        int rowSkip = 0;
                        var node = adj.rows.First;
                        while (node != null && node.Value < i)
                        {
                            node = node.Next;
                            rowSkip++;
                        }

                        if (node != null && node.Value == i)
                            continue;

                        LinkedListNode<int> rowNode, colNode;

                        int adjCol = adj.smallestAdjugatableCol, skipCol = adj.skipAdjCol;
                        adj.Add(i, adjCol, out rowNode, out colNode);
                        result += Cofactor(matrix, adj, NegOneRaiseTo(i - rowSkip + adjCol - skipCol)) * matrix.Get(i, adjCol);
                        adj.Remove(rowNode, colNode);
                    }

                    return result * multiplier;
                }

                public static double Cofactor(ISquareMatrix matrix, int row, int col)
                {
                    if (matrix.dim == 2)
                        return IMatrix.NegOneRaiseTo(row + col) * matrix.Get(1 - row, 1 - col);

                    return ISquareMatrix.Cofactor(matrix, new ISquareMatrix.AdjugateSum(row, col), IMatrix.NegOneRaiseTo(row + col));
                }

                public struct AdjugateSum
                {
                    public LinkedList<int> rows, cols;

                    public int smallestAdjugatableRow { get; private set; } = 0;
                    public int smallestAdjugatableCol { get; private set; } = 0;
                    public int skipAdjCol { get; private set; } = 0;
                    public int rowSum { get; private set; } = 0;
                    public int colSum { get; private set; } = 0;
                    public int Count => rows.Count;

                    public AdjugateSum()
                    {
                        rows = new LinkedList<int>();
                        cols = new LinkedList<int>();
                    }

                    public AdjugateSum(int row, int col)
                    {
                        rows = new LinkedList<int>();
                        cols = new LinkedList<int>();

                        Add(row, col);
                    }

                    public void UpdateColSkip()
                    {
                        skipAdjCol = 0;
                        var node = cols.First;
                        while (node != null && node.Value < smallestAdjugatableCol)
                        {
                            node = node.Next;
                            skipAdjCol++;
                        }
                    }

                    public void Remove(LinkedListNode<int> rowNode, LinkedListNode<int> colNode)
                    {
                        rowSum -= rowNode.Value;
                        colSum -= colNode.Value;

                        if (rowNode.Value < smallestAdjugatableRow)
                            smallestAdjugatableRow = rowNode.Value;

                        if (colNode.Value < smallestAdjugatableCol)
                            smallestAdjugatableCol = colNode.Value;

                        rows.Remove(rowNode);
                        cols.Remove(colNode);
                        UpdateColSkip();
                    }

                    public void Add(int row, int col)
                    {
                        LinkedListNode<int> rowNode, colNode;
                        Add(row, col, out rowNode, out colNode);
                    }

                    public void Add(int row, int col, out LinkedListNode<int> rowNode, out LinkedListNode<int> colNode)
                    {
                        LinkedListNode<int> node;
                        rowNode = null;
                        colNode = null;

                        bool added = false;
                        node = rows.First;
                        for (; node != null; node = node.Next)
                        {
                            if (node.Value > row)
                            {
                                rowSum += row;
                                rowNode = rows.AddBefore(node, row);
                                added = true;

                                if (smallestAdjugatableRow < row)
                                    break;
                                row++;

                                while (node != null && node.Value == row)
                                {
                                    node = node.Next;
                                    row++;
                                }
                                smallestAdjugatableRow = row;
                                break;

                            }
                            else if (row == node.Value)
                                break;
                        }

                        if (!added)
                        {
                            rowSum += row;
                            rowNode = rows.AddLast(row);

                            if (smallestAdjugatableRow >= row)
                                smallestAdjugatableRow = row + 1;
                        }

                        added = false;
                        node = cols.First;
                        for (; node != null; node = node.Next)
                        {
                            if (node.Value > col)
                            {
                                colSum += col;
                                colNode = cols.AddBefore(node, col);
                                added = true;

                                if (smallestAdjugatableCol < col)
                                    break;
                                col++;

                                while (node != null && node.Value == col)
                                {
                                    node = node.Next;
                                    col++;
                                }
                                smallestAdjugatableCol = col;
                                break;

                            }
                            else if (col == node.Value)
                                break;
                        }

                        if (!added)
                        {
                            colSum += col;
                            colNode = cols.AddLast(col);

                            if (smallestAdjugatableCol >= col)
                                smallestAdjugatableCol = col + 1;
                        }
                        UpdateColSkip();
                    }
                }

                public static ISquareMatrix operator *(ISquareMatrix A, ISquareMatrix B) => A.Multiply(B).ToSquare;

                public static ISquareMatrix operator +(ISquareMatrix matrix, double value) => (ISquareMatrix)matrix.Add(value);
                public static ISquareMatrix operator -(ISquareMatrix matrix, double value) => (ISquareMatrix)matrix.Subtract(value);
                public static ISquareMatrix operator *(ISquareMatrix matrix, double value) => (ISquareMatrix)matrix.Multiply(value);
                public static ISquareMatrix operator /(ISquareMatrix matrix, double value) => (ISquareMatrix)matrix.Divide(value);

                public static ISquareMatrix operator +(double value, ISquareMatrix matrix) => (ISquareMatrix)matrix.Add(value);
                public static ISquareMatrix operator -(double value, ISquareMatrix matrix) => (ISquareMatrix)matrix.LeftSubtract(value);
                public static ISquareMatrix operator *(double value, ISquareMatrix matrix) => (ISquareMatrix)matrix.Multiply(value);
            }

            public class DenseMatrix : IMatrix
            {
                public double[][] content;
                public int rowCount => content.Length;
                public int colCount => content[0].Length;

                public virtual IMatrix Transpose
                {
                    get
                    {
                        IMatrix result = InstanceT();

                        for (int i = 0; i < colCount; i++)
                            for (int j = 0; j < rowCount; j++)
                                result.Set(i, j, Get(j, i));

                        return result;
                    }
                }

                public virtual ISquareMatrix ToSquare
                {
                    get
                    {
                        if (this is ISquareMatrix)
                            return (ISquareMatrix)Clone();

                        ISquareMatrix result = new DenseSquareMatrix(rowCount);
                        result.SetTo(this);
                        return result;
                    }
                }

                public DenseMatrix(int row, int col)
                {
                    content = new double[row][];

                    for (int i = 0; i < row; i++)
                        content[i] = new double[col];
                }

                public DenseMatrix(double[][] content)
                {
                    this.content = content;
                }

                public virtual IMatrix Instance() => new DenseMatrix(rowCount, colCount);

                public virtual IMatrix InstanceT() => new DenseMatrix(colCount, rowCount);

                public virtual double Get(int row, int col) => content[row][col];

                public virtual bool Set(int row, int col, double value)
                {
                    content[row][col] = value;
                    return true;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            Set(i, j, matrix.Get(i, j));

                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix result = Instance();
                    result.SetTo(this);

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (colCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[rowCount];

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result[i] += Get(i, j) * vector.content[j];

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (rowCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[colCount];

                    for (int i = 0; i < colCount; i++)
                        for (int j = 0; j < rowCount; j++)
                            result[i] += Get(i, j) * vector.content[j];

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    for (int row = 0; row < rowCount; row++)
                        for (int col = 0; col < matrix.colCount; col++)
                            for (int i = 0; i < colCount; i++)
                                result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));

                    return result;
                }

                public virtual IMatrix Add(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(double value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) * value);

                    return result;
                }

                public virtual IMatrix Divide(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) / value);

                    return result;
                }
            }

            public class DenseSquareMatrix : DenseMatrix, ISquareMatrix
            {
                public int dim => content.Length;

                public DenseSquareMatrix(int dim) : base(dim, dim) { }

                public DenseSquareMatrix(double[][] content) : base(content)
                {
                    if (content.Length != content[0].Length)
                        throw new Exception("Invalid square matrix content");
                }

                public override IMatrix Instance() => new DenseSquareMatrix(dim);
                public override IMatrix InstanceT() => new DenseSquareMatrix(dim);

                public virtual ISquareMatrix Invert()
                {
                    return (ISquareMatrix)Adjugate().Divide(Determinant());
                }

                public virtual double Determinant()
                {
                    if (dim == 1)
                        return content[0][0];
                    else if (dim == 2)
                        return content[0][0] * content[1][1] - content[1][0] * content[0][1];
                    else
                    {
                        double result = 0;

                        for (int i = 0; i < dim; i++)
                            result += Cofactor(i, 0) * content[i][0];

                        return result;
                    }
                }

                public virtual double Cofactor(int row, int col)
                {
                    return ISquareMatrix.Cofactor(this, row, col);
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, Cofactor(j, i));

                    return result;
                }
            }

            public class DiagonalMatrix : ISquareMatrix
            {
                public double[] content;

                public int dim => content.Length;
                public int rowCount => content.Length;
                public int colCount => content.Length;

                public IMatrix Transpose => Clone();

                public ISquareMatrix ToSquare => (ISquareMatrix)Clone();

                public DiagonalMatrix(int dim)
                {
                    content = new double[dim];
                }

                public DiagonalMatrix(double[] content)
                {
                    this.content = content;
                }

                public virtual IMatrix Instance() => new DiagonalMatrix(dim);
                public virtual IMatrix InstanceT() => Instance();

                public virtual double Get(int row, int col) 
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    return row == col ? content[row] : 0;
                }

                public virtual bool Set(int row, int col, double value)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if (row != col)
                        return false;

                    content[row] = value;
                    return true;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    if (!(matrix is DiagonalMatrix) || matrix.rowCount != dim)
                        return false;

                    for (int i = 0; i < dim; i++)
                        Set(i, i, matrix.Get(i, i));
                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix matrix = Instance();
                    matrix.SetTo(this);
                    return matrix;
                }

                public virtual ISquareMatrix Invert()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, 1 / Get(i, i));

                    return result;
                }

                public virtual double Determinant()
                {
                    double result = 1;
                    for (int i = 0; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual double Cofactor(int row, int col)
                {
                    if (row != col)
                        return 0;

                    double result = 1;

                    for (int i = 0; i < row; i++)
                        result *= Get(i, i);
                    for (int i = row + 1; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();
                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Cofactor(i, i));

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (colCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[rowCount];

                    for (int i = 0; i < dim; i++)
                        result[i] += Get(i, i) * vector.content[i];

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (rowCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[colCount];

                    for (int i = 0; i < dim; i++)
                        result[i] += Get(i, i) * vector.content[i];

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    for (int row = 0; row < rowCount; row++)
                        for (int col = 0; col < matrix.colCount; col++)
                            result.Set(row, col, Get(row, col) * matrix.Get(col, col));

                    return result;
                }

                public virtual IMatrix Add(double value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(double value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(double value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Get(i, i) * value);

                    return result;
                }

                public virtual IMatrix Divide(double value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Get(i, i) / value);

                    return result;
                }
            }

            public class TriangularMatrix : ISquareMatrix
            {
                public double[] content;
                public bool isUpper;

                public int dim { get; protected set; }
                public int rowCount => dim;
                public int colCount => dim;

                public virtual IMatrix Transpose {
                    get
                    {
                        TriangularMatrix result = (TriangularMatrix)Clone();
                        result.isUpper = !isUpper;
                        return result;
                    }
                }

                public ISquareMatrix ToSquare => (ISquareMatrix)Clone();

                public TriangularMatrix(int dim, bool isUpper)
                {
                    content = new double[(dim * (1 + dim)) >> 1];
                    this.isUpper = isUpper;
                    this.dim = dim;
                }

                public TriangularMatrix(double[] content, bool isUpper)
                {
                    this.content = content;
                    this.isUpper = isUpper;
                    this.dim = (int)(MathF.Sqrt(1 + 8 * dim) - 1) >> 1;
                }

                public virtual IMatrix Instance() => new TriangularMatrix(dim, isUpper);
                public virtual IMatrix InstanceT() => new TriangularMatrix(dim, !isUpper);

                public virtual double Get(int row, int col)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if ((row > col) == isUpper && row != col) 
                        return 0;
                    
                    if (isUpper)
                        return content[((col * (1 + col)) >> 1) + row];
                    else
                        return content[((row * (1 + row)) >> 1) + col];
                }

                public virtual bool Set(int row, int col, double value)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if ((row > col) == isUpper && row != col)
                        return false;

                    if (isUpper)
                        content[((col * (1 + col)) >> 1) + row] = value;
                    else
                        content[((row * (1 + row)) >> 1) + col] = value;

                    return true;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    if (!(matrix is DiagonalMatrix || matrix is TriangularMatrix) || matrix.rowCount != dim)
                        return false;

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                Set(i, j, matrix.Get(i, j));
                    } 
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                Set(i, j, matrix.Get(i, j));
                    }

                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix matrix = Instance();
                    matrix.SetTo(this);
                    return matrix;
                }

                public virtual ISquareMatrix Invert()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();
                    double det = Determinant();

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i) / det);
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i) / det);
                    }

                    return result;
                }

                public virtual double Determinant()
                {
                    double result = 1;
                    for (int i = 0; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual double Cofactor(int row, int col)
                {
                    if (row > col != isUpper)
                        return 0;

                    if (row == col)
                        return Determinant() / Get(row, col);

                    return ISquareMatrix.Cofactor(this, row, col);
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i));
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i));
                    }

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (dim != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[dim];

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result[i] += Get(i, j) * vector.content[j];
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result[i] += Get(i, j) * vector.content[j];
                    }

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (dim != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[dim];

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result[j] += Get(i, j) * vector.content[i];
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result[j] += Get(i, j) * vector.content[i];
                    }

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    if (isUpper)
                    {
                        for (int row = 0; row < rowCount; row++)
                            for (int col = 0; col < matrix.colCount; col++)
                                for (int i = row; i < colCount; i++)
                                    result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));
                    }
                    else
                    {
                        for (int row = 0; row < rowCount; row++)
                            for (int col = 0; col < matrix.colCount; col++)
                                for (int i = 0; i <= row; i++)
                                    result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));
                    }

                    return result;
                }

                public virtual IMatrix Add(double value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(double value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(double value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(double value)
                {
                    TriangularMatrix result = (TriangularMatrix)Instance();

                    for(int i = 0; i < content.Length; i++)
                        result.content[i] = content[i] * value;

                    return result;
                }

                public virtual IMatrix Divide(double value)
                {
                    TriangularMatrix result = (TriangularMatrix)Instance();

                    for (int i = 0; i < content.Length; i++)
                        result.content[i] = content[i] / value;

                    return result;
                }
            }

            public class Vector
            {
                public double[] content;

                public int dim => content.Length;

                public Vector(double[] content)
                {
                    this.content = content;
                } 

                public Vector(int size)
                {
                    this.content = new double[size];
                }

                public void SetTo(Vector vector)
                {
                    for (int i = 0; i < dim; i++)
                        content[i] = vector.content[i];
                }

                public Vector Clone()
                {
                    Vector clone = new Vector(dim);

                    for (int i = 0; i < dim; i++)
                        clone.content[i] = content[i];

                    return clone;
                }

                public static Vector Clone(double[] content)
                {
                    Vector clone = new Vector(content.Length);

                    for (int i = 0; i < clone.dim; i++)
                        clone.content[i] = content[i];

                    return clone;
                }

                public static double[] Add(double[] a, double[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b[i];

                    return result;
                }

                public static double[] Add(double[] a, double b)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b;

                    return result;
                }

                public static double[] Add(double b, double[] a)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b;

                    return result;
                }

                public static double[] Subtract(double[] a, double[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] - b[i];

                    return result;
                }

                public static double[] Subtract(double[] a, double b)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] - b;

                    return result;
                }

                public static double[] Subtract(double b, double[] a)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = b - a[i];

                    return result;
                }

                public static double[] Multiply(double[] a, double b)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] * b;

                    return result;
                }

                public static double[] Divide(double[] a, double b)
                {
                    double[] result = new double[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] / b;

                    return result;
                }

                public static double Dot(double[] a, double[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    double result = 0;

                    for (int i = 0; i < a.LongLength; i++)
                        result += a[i] * b[i];

                    return result;
                }

                public static double Dot(Vector a, Vector b) => Dot(a.content, b.content);

                public static implicit operator Vector(double[] content) => new Vector(content);

                public static Vector operator +(Vector a, Vector b) => Add(a.content, b.content);
                public static Vector operator +(Vector a, double b) => Add(a.content, b);
                public static Vector operator +(double b, Vector a) => Add(a.content, b);

                public static Vector operator -(Vector a, Vector b) => Subtract(a.content, b.content);
                public static Vector operator -(Vector a, double b) => Subtract(a.content, b);
                public static Vector operator -(double b, Vector a) => Subtract(b, a.content);

                public static Vector operator *(Vector a, double b) => Multiply(a.content, b);
                public static Vector operator *(double b, Vector a) => Multiply(a.content, b);
                public static Vector operator /(Vector a, double b) => Divide(a.content, b);
            }
        }
    }
}