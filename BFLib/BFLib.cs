using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;

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
                public static double[] CGMethod(SquareMatrix A, Vector b, double epsilon = 1E-6)
                {
                    if (A.dim != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");

                    SquareMatrix matrixA = SquareMatrix.Clone(A.content);
                    Vector
                        result = new double[A.dim],
                        residual = Vector.Clone(b.content), preResidual = Vector.Clone(b.content),
                        direction = Vector.Clone(residual.content);

                    if (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * matrixA, direction);

                        result += alpha * direction;
                        residual -= alpha * matrixA * direction;
                    }

                    while (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double beta = Vector.Dot(residual, residual) / Vector.Dot(preResidual, preResidual);

                        direction = residual + beta * direction;

                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * matrixA, direction);

                        preResidual.SetTo(residual);
                        result += alpha * direction;
                        residual -= alpha * matrixA * direction;
                    }

                    return result.content;
                }

                public static double[] RawCGMethod(SquareMatrix A, Vector b, SquareMatrix inverseFactorM, double epsilon = 1E-6)
                {
                    if (A.dim != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");

                    SquareMatrix matrixA = SquareMatrix.Clone(A.content);
                    Vector
                        result = new double[A.dim],
                        residual = Vector.Clone(b.content), preResidual = Vector.Clone(b.content),
                        direction = Vector.Clone(residual.content);

                    if (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * matrixA, direction);

                        result += alpha * direction;
                        residual -= alpha * matrixA * direction;
                    }

                    while (Vector.Dot(residual, residual) > epsilon * residual.dim)
                    {
                        double beta = Vector.Dot(residual, residual) / Vector.Dot(preResidual, preResidual);

                        direction = residual + beta * direction;

                        double alpha = Vector.Dot(residual, residual) / Vector.Dot(direction * matrixA, direction);

                        preResidual.SetTo(residual);
                        result += alpha * direction;
                        residual -= alpha * matrixA * direction;
                    }

                    return result.content;
                }

                /// <returns>A tuple of a lower matrix and an upper matrix</returns>
                public static (SquareMatrix, SquareMatrix) LUFac(SquareMatrix A, double epsilon = 1E-3)
                {
                    SquareMatrix lower = new SquareMatrix(A.dim);
                    SquareMatrix upper = new SquareMatrix(A.dim);

                    for (int i = 0; i < A.dim; i++)
                        for (int j = 0; j < i + 1; j++)
                        {
                            // Row iterate
                            // j : row index
                            double rowSum = 0;
                            for (int k = 0; k < j; k++)
                                rowSum += lower.content[j][k] * upper.content[k][i];
                            upper.content[j][i] = A.content[j][i] - rowSum;

                            // Column iterate
                            // j : column index
                            if (i == j)
                                lower.content[i][j] = 1;
                            else
                            {
                                double colSum = 0;
                                for (int k = 0; k < j; k++)
                                    colSum += lower.content[i][k] * upper.content[k][j];

                                lower.content[i][j] = (A.content[i][j] - colSum) / upper.content[j][j];
                            }

                        }

                    return (lower, upper);
                }

                public static SquareMatrix IncompleteCholeskyFac(SquareMatrix A, double epsilon = 1E-3)
                {
                    SquareMatrix result = new SquareMatrix(A.dim);

                    for (int row = 0; row < A.dim; row++)
                        for (int col = 0; col < row + 1; col++)
                        {
                            if (A.content[row][col] < epsilon)
                            {
                                result.content[row][col] = 0;
                                continue;
                            }

                            double sum = 0;
                            for (int i = 0; i < col; i++)
                                sum += result.content[row][i] * result.content[col][i];

                            if (col == row)
                                result.content[row][col] = Math.Sqrt(A.content[row][col] - sum);
                            else
                                result.content[row][col] = (A.content[row][col] - sum) / result.content[col][col];
                        }

                    return result;
                }
            }
            public class SquareMatrix
            {
                public double[][] content;

                public int dim => content.Length;

                public SquareMatrix Transpose
                {
                    get
                    {
                        SquareMatrix result = new SquareMatrix(dim);

                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j < dim; j++)
                                result.content[i][j] = content[j][i];

                        return result;
                    }
                }

                public SquareMatrix(int dim)
                {
                    this.content = new double[dim][];

                    for (int i = 0; i < dim; i++)
                        content[i] = new double[dim];
                }

                public SquareMatrix(double[][] content)
                {
                    if (content.Length != content[0].Length)
                        throw new Exception("Invalid square matrix content");

                    this.content = content;
                }

                public static SquareMatrix Identity(int dim)
                {
                    SquareMatrix identity = new SquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        identity.content[i][i] = 1;

                    return identity;
                }

                public static SquareMatrix Diag(params double[] nums)
                {
                    SquareMatrix matrix = new SquareMatrix(nums.Length);

                    for (int i = 0; i < nums.Length; i++)
                        matrix.content[i][i] = nums[i];

                    return matrix;
                }

                public SquareMatrix Invert()
                {
                    return AdjugateMatrix() / Determinant();
                }

                public double Determinant()
                {
                    if (dim == 1)
                        return content[0][0];
                    else if (dim == 2)
                        return content[0][0] * content[1][1] - content[1][0] * content[0][1];
                    else {
                        double result = 0;

                        for (int i = 0; i < dim; i++)
                            result += Adjugate(i, 0) * content[i][0];

                        return result;
                    }
                }

                public double Adjugate(int row, int col)
                {
                    if (dim == 2)
                        return NegOneRaiseTo(row + col) * content[1 - row][1 - col];
                    
                    return Adjugate(new AdjugateSum(row, col), NegOneRaiseTo(row + col));
                }

                private double Adjugate(AdjugateSum adj, double multiplier, bool coefMultiply = false)
                {
                    if (dim - adj.Count == 2)
                    {
                        int sum = (dim - 1) * dim / 2;
                        int row1 = adj.smallestAdjugatableRow, row2 = sum - adj.rowSum - row1,
                            col1 = adj.smallestAdjugatableCol, col2 = sum - adj.colSum - col1;

                        return multiplier * (content[row1][col1] * content[row2][col2] - content[row1][col2] * content[row2][col1]);
                    }

                    double result = 0;

                    for (int i = 0; i < dim; i++)
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
                        result += Adjugate(adj, NegOneRaiseTo(i - rowSkip + adjCol - skipCol)) * content[i][adjCol];
                        adj.Remove(rowNode, colNode);
                    }

                    return result * multiplier;
                }

                public SquareMatrix AdjugateMatrix()
                {
                    SquareMatrix result = new SquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.content[i][j] = Adjugate(j, i);

                    return result;
                }

                private int NegOneRaiseTo(int num)
                {
                    return num % 2 == 0 ? 1 : -1;
                }

                public void SetTo(SquareMatrix matrix)
                {
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            content[i][j] = matrix.content[i][j];
                }

                public SquareMatrix Clone()
                {
                    SquareMatrix clone = new SquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            clone.content[i][j] = content[i][j];

                    return clone;
                }

                public static SquareMatrix Clone(double[][] content)
                {
                    if (content.Length != content[0].Length)
                        throw new Exception("Invalid square matrix content");

                    SquareMatrix clone = new SquareMatrix(content.Length);

                    for (int i = 0; i < clone.dim; i++)
                        for (int j = 0; j < clone.dim; j++)
                            clone.content[i][j] = content[i][j];

                    return clone;
                }

                public static double[] Multiply(double[][] matrix, double[] vector)
                {
                    if (matrix[0].Length != vector.LongLength)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[matrix.Length];

                    for (int i = 0; i < matrix.Length; i++)
                        for (int j = 0; j < matrix[0].Length; j++)
                            result[i] += matrix[i][j] * vector[j];

                    return result;
                }

                public static double[] Multiply(double[] vector, double[][] matrix)
                {
                    if (matrix.Length != vector.LongLength)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[] result = new double[matrix[0].Length];

                    for (int i = 0; i < matrix[0].Length; i++)
                        for (int j = 0; j < matrix.Length; j++)
                            result[i] += matrix[i][j] * vector[j];

                    return result;
                }

                public static double[][] Multiply(double[][] a, double[][] b)
                {
                    if (a[0].Length != b.Length)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    double[][] result = new double[a.Length][];
                    for (int i = 0; i < a.Length; i++)
                        result[i] = new double[a[0].Length];

                    for (int row = 0; row < a.Length; row++)
                        for (int col = 0; col < a.Length; col++)
                            for (int i = 0; i < a[0].Length; i++)
                                result[row][col] += a[row][i] * b[i][col];

                    return result;
                }

                public static double[][] Add(double[][] a, double b)
                {
                    double[][] result = new double[a.LongLength][];

                    for (int i = 0; i < a.Length; i++)
                    {
                        result[i] = new double[a[0].Length];
                        for (int j = 0; j < a[0].Length; j++)
                            result[i][j] = a[i][j] + b;
                    }

                    return result;
                }

                public static double[][] Subtract(double[][] a, double b)
                {
                    return Add(a, -b);
                }
                public static double[][] Subtract(double b, double[][] a)
                {
                    double[][] result = new double[a.LongLength][];

                    for (int i = 0; i < a.Length; i++)
                    {
                        result[i] = new double[a[0].Length];
                        for (int j = 0; j < a[0].Length; j++)
                            result[i][j] = b - a[i][j];
                    }

                    return result;
                }

                public static double[][] Multiply(double[][] a, double b)
                {
                    double[][] result = new double[a.LongLength][];

                    for (int i = 0; i < a.Length; i++)
                    {
                        result[i] = new double[a[0].Length];
                        for (int j = 0; j < a[0].Length; j++)
                            result[i][j] = a[i][j] * b;
                    }

                    return result;
                }

                public static double[][] Divide(double[][] a, double b)
                {
                    double[][] result = new double[a.LongLength][];

                    for (int i = 0; i < a.Length; i++)
                    {
                        result[i] = new double[a[0].Length];
                        for (int j = 0; j < a[0].Length; j++)
                            result[i][j] = a[i][j] / b;
                    }

                    return result;
                }

                public static implicit operator SquareMatrix(double[][] content) => new SquareMatrix(content);

                public static SquareMatrix operator *(SquareMatrix left, SquareMatrix right) => Multiply(left.content, right.content);
                public static Vector operator *(SquareMatrix left, Vector right) => Multiply(left.content, right.content);
                public static Vector operator *(Vector left, SquareMatrix right) => Multiply(left.content, right.content);

                public static SquareMatrix operator +(SquareMatrix left, double right) => Add(left.content, right);
                public static SquareMatrix operator -(SquareMatrix left, double right) => Subtract(left.content, right);
                public static SquareMatrix operator *(SquareMatrix left, double right) => Multiply(left.content, right);
                public static SquareMatrix operator /(SquareMatrix left, double right) => Divide(left.content, right);

                public static SquareMatrix operator +(double right, SquareMatrix left) => Add(left.content, right);
                public static SquareMatrix operator -(double right, SquareMatrix left) => Subtract(right, left.content);
                public static SquareMatrix operator *(double right, SquareMatrix left) => Multiply(left.content, right);

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