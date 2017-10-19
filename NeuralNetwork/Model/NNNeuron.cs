using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNNeuron
    {
        private int numberOfInputs;
        public double[] inputs { get; private set; }
        public double[] weights { get; private set; }
        public double w0 { get; set; }
        public double output { get; private set; }
        public double delta { get; set; }

        public NNNeuron()
        {
            inputs = null;
        }

        public int SetInputs(ref double[] inputs)
        {
            if (inputs.Length == numberOfInputs)
            {
                this.inputs = inputs;
                return 0;
            } else
            {
                return 1;
            }
        }

        public void Initialize(int numberOfInputs)
        {
            this.numberOfInputs = numberOfInputs;
            this.weights = new double[numberOfInputs];
            var Rand = new Random(Guid.NewGuid().GetHashCode());
            w0 = (Rand.NextDouble() - 0.5) * 6;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (Rand.NextDouble() - 0.5) * 6;
            } 
        }

        
        public double Activation()
        {
            double sum = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * weights[i];
            }
            sum += w0;

            output = Transfert(sum);
            return output;
        }

        private double Transfert(double s)
        {
            return 1 / (1 + Math.Exp(-s));
        }

        static public double TransfertDerivative(double s)
        {
            return s * (1 - s);
        }


    }
}
