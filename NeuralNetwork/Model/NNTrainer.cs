using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNTrainer
    {

        public NNTrainer()
        {
        }

        public int Train(NNNeuralNetwork nn, double[][] dataset, double[][] expectedResults, double learningRate, int nbEpoch)
        {
            /*
            if (dataset.GetLength(0) != nn.nbOfInputs || expectedResults.GetLength(0) != nn.nbOfOutputs)
            {
                return -1;
            }
            */

            for (int i = 0; i < nbEpoch; i++)
            {
                double sumError = 0.0;
                for (int j = 0; j < dataset.Length; j++)
                {
                    double[] inputs = dataset[j];
                    double[] outputs = nn.ProcessInputs(ref inputs);
                    double[] expected = expectedResults[j];

                    /*
                    var strBuilder = new StringBuilder();
                    strBuilder.Append("Line=");
                    strBuilder.Append(j);
                    strBuilder.Append(" : inputs=");
                    for (int k = 0; k < inputs.Length-1; k++)
                    {
                        strBuilder.Append(inputs[k]);
                        strBuilder.Append(",");
                    }
                    strBuilder.Append(inputs[inputs.Length-1]);

                    strBuilder.Append(" ; outputs=");
                    for (int k = 0; k < outputs.Length - 1; k++)
                    {
                        strBuilder.Append(outputs[k]);
                        strBuilder.Append(",");
                    }
                    strBuilder.Append(outputs[outputs.Length - 1]);

                    strBuilder.Append(" ; expected=");
                    for (int k = 0; k < expected.Length - 1; k++)
                    {
                        strBuilder.Append(expected[k]);
                        strBuilder.Append(",");
                    }
                    strBuilder.Append(expected[expected.Length - 1]);

                    Console.WriteLine(strBuilder.ToString());
                    */
                    for (int k = 0; k < outputs.Length; k++)
                    {
                        sumError += (expected[k] - outputs[k]);
                    }
                    BackwardPropagateError(nn, expected);
                    UpdateWeights(nn, inputs, learningRate);
                }
                Console.WriteLine(">Epoch={0}, learningRate={1}, error={2}", i, learningRate, sumError);
                Console.WriteLine("---------------------------------------");
            }

            return 0;
        }

        private int BackwardPropagateError(NNNeuralNetwork nn, double[] expected)
        {
            if (nn.nbOfOutputs == expected.Length)
            {
                // Parcours inverse des layers
                for (int i = nn.nbOfHiddenLayers; i >= 0; i--)
                {
                    NNLayer Layer = nn.Layers[i];
                    double[] errors = new double[Layer.nbOfNeurons];

                    // Si ce n'est pas le dernier layer
                    if (i != nn.nbOfHiddenLayers)
                    {
                        // Parcours des neurones du layer courant
                        // Calcul de l'erreur
                        for (int j = 0; j < Layer.nbOfNeurons; j++)
                        {
                            var error = 0.0;
                            // Parcours des neurones du layers précédent
                            nn.Layers[i + 1].Neurons.ForEach(Neuron => {
                                error += Neuron.weights[j] * Neuron.delta;
                            });
                            errors[i] = error;
                        }
                    }
                    // Si c'est le dernier layer
                    else
                    {
                        // Parcours des neurones du layer courant
                        // Calcul de l'erreur
                        for (int j = 0; j < Layer.nbOfNeurons; j++)
                        {
                            errors[j] = (expected[j] - Layer.Neurons[j].output);
                        }
                    }

                    // Parcours des neurones du layer courant
                    // Calcul de delta
                    for (int j = 0; j < Layer.nbOfNeurons; j++)
                    {
                        NNNeuron Neuron = Layer.Neurons[j];
                        Neuron.delta = errors[j] * NNNeuron.TransfertDerivative(Neuron.output);
                    }
                }

                return 0;
            }
            else
            {
                return 1;
            }
        }

        private void UpdateWeights(NNNeuralNetwork nn, double[] nnInputs, double learningRate)
        {
            for (int i = 0; i < nn.nbOfHiddenLayers + 1; i++)
            {
                NNLayer Layer = nn.Layers[i];
                double[] inputs;

                // Récupération des inputs du layer
                if (i != 0)
                {
                    inputs = new double[nn.Layers[i - 1].nbOfNeurons];
                    for (int j = 0; j < nn.Layers[i - 1].nbOfNeurons; j++)
                    {
                        inputs[j] = nn.Layers[i - 1].Neurons[j].output;
                    }
                }
                else
                    inputs = nnInputs;

                // Mise à jour des poids de chaque neurone du layer
                Layer.Neurons.ForEach(Neuron =>
                {
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        Neuron.weights[j] += learningRate * Neuron.delta * inputs[j];
                    }
                    Neuron.w0 += learningRate * Neuron.delta;
                });
            }
        }



    }
}
