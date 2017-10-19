using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNLayer
    {
        private int nbOfInputs;
        public int nbOfNeurons { get; private set; }
        public double[] inputs;
        public List<NNNeuron> Neurons { get; private set; }
        
        public NNLayer(int NbNodes)
        {
            this.nbOfNeurons = NbNodes;
            this.Neurons = new List<NNNeuron>(nbOfNeurons);
        }

        public int SetInputs(ref double[] inputs)
        {
            if (inputs.Length == nbOfInputs)
            {
                this.inputs = inputs;
                this.nbOfInputs = inputs.Length;
                return 0;
            } 
            else
            {
                return -1;
            }
        }

        public void Initialize(int NbInputs)
        {
            nbOfInputs = NbInputs;
            Neurons.Clear();
            for (var i = 0; i < nbOfNeurons; i++)
            {
                var Node = new NNNeuron();
                Node.Initialize(nbOfInputs);
                Neurons.Add(Node);
            }
        }

        public double[] ProcessInputs()
        {
            double[] outputs = new double[nbOfNeurons];
            for (var i = 0; i < nbOfNeurons; i++)
            {
                Neurons[i].SetInputs(ref inputs);
                outputs[i] = Neurons[i].Activation();
            }
            return outputs;
        }

    }
}
