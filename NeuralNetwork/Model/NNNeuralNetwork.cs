using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNNeuralNetwork
    {
        public int nbOfInputs { get; private set; }
        public int nbOfOutputs { get; private set; }
        public int nbOfHiddenLayers { get; private set; }
        public int[] sizeHiddenLayers { get; private set; }

        public List<NNLayer> Layers { get; private set; }

        public NNNeuralNetwork(int nbInputs, int nbOutputs, int[] sizeHL)
        {
            this.nbOfInputs = nbInputs;
            this.nbOfOutputs = nbOutputs;

            if(sizeHL == null)
            {
                this.nbOfHiddenLayers = 0;
            }
            else
            {
                this.nbOfHiddenLayers = sizeHL.Length;
            }
            this.sizeHiddenLayers = sizeHL;

            Layers = new List<NNLayer>(nbOfHiddenLayers);

            if (sizeHiddenLayers.Length == nbOfHiddenLayers)
            {
                for (var i = 0; i < nbOfHiddenLayers; i++)
                {
                    var HiddenLayer = new NNLayer(sizeHiddenLayers[i]);
                    Layers.Add(HiddenLayer);
                }
                var OutputLayer = new NNLayer(nbOfOutputs);
                Layers.Add(OutputLayer);
            }
            
        }

        public void Initialise()
        {
            for (var i = 0; i < Layers.Count; i++)
            {
                if (i == 0)
                {
                    Layers[i].Initialize(nbOfInputs);
                }
                else
                {
                    Layers[i].Initialize(Layers[i - 1].nbOfNeurons);
                }
            }
        }

        public double[] ProcessInputs(ref double[] inputs)
        {
            if(inputs.Length != nbOfInputs)
            {
                return null;
            }

            double[] previousOutputs = inputs;
            var errorOccurred = false;

            for (var i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].SetInputs(ref previousOutputs) == 0)
                {
                    previousOutputs = Layers[i].ProcessInputs();
                }
                else
                {
                    errorOccurred = true;
                    break;
                }
            }

            if (errorOccurred)
                return null;
            else
                return previousOutputs; 
        }


    }
}
