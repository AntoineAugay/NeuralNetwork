using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Model
{
    class NNManager
    {
        public NNManager()
        {
            int NumberOfInputs = 2;



            NNNode Node = new NNNode();
            Node.Initialize(NumberOfInputs);
            NNTrainerOR Trainer = new NNTrainerOR(100000);
            Random Rand = new Random(Guid.NewGuid().GetHashCode());
            Trainer.Train(Node);

            for (int i = 0; i < 50; i++)
            {
                int input0 = Rand.Next(2);
                int input1 = Rand.Next(2);
                Node.inputs[0] = input0;
                Node.inputs[1] = input1;

                int expected = input0 + input1;
                if (expected == 2) expected = 1;

                double result = Node.ProcessInputs();

                Console.WriteLine("Line {0} : Expected {1}, Real {2} ", i, expected, result);
            }
        }


    }
}
