using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNLayer
    {
        private int NbOfInputs;
        private int NbOfNodes;
        public double[] inputs;
        public List<NNNode> Nodes { get; private set; }
        
        public NNLayer(int NbNodes)
        {
            this.NbOfNodes = NbNodes;
            this.Nodes = new List<NNNode>(NbOfNodes);
        }

        public void SetInputs(ref double[] inputs)
        {
            this.inputs = inputs;
        }

        public void Initialize(int NbInputs)
        {
            NbOfInputs = NbInputs;
            Nodes.Clear();
            for (int i = 0; i < NbOfNodes; i++)
            {
                NNNode Node = new NNNode();
                Node.SetInputs(ref inputs);
                Node.Initialize(NbOfInputs);
                Nodes.Add(Node);
            }
        }

        public void ProcessInputs()
        {
            double[] outputs = new double[NbOfNodes];
            for (int i = 0; i < NbOfNodes; i++)
            {
                Nodes[i].SetInputs(ref inputs);
                outputs[i] = Nodes[i].ProcessInputs();
            }
        }

    }
}
