using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNNode
    {
        private int NumberOfInputs;
        public double[] inputs { get; private set; }
        public double[] weights { get; private set; }
        public double w0 { get; set; }

        public NNNode()
        {
            inputs = null;
        }

        public int SetInputs(ref double[] inputs)
        {
            if (inputs.Length == NumberOfInputs)
            {
                this.inputs = inputs;
                return 0;
            } else
            {
                return 1;
            }
        }

        public void Initialize(int NumberOfInputs)
        {
            this.weights = new double[NumberOfInputs];
            Random Rand = new Random(Guid.NewGuid().GetHashCode());
            w0 = (Rand.NextDouble() - 0.5) * 6;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (Rand.NextDouble() - 0.5) * 6;
            } 
        }

        public double ProcessInputs()
        {
            double sum = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * weights[i];
            }
            sum += w0;
            return ActivationFonction(sum);
        }

        private double ActivationFonction(double s)
        {
            return 1 / (1 + Math.Exp(-s));
        }
    }
}
