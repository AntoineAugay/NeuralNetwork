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
            const int nbOfInputs = 2;
            const int nbOfOutputs = 1;
            const int sizeDataSet = 10000;
            var sizeLayers = new int[0];
            //sizeLayers[0] = 2;
            var Rand = new Random(Guid.NewGuid().GetHashCode());

            var nn = new NNNeuralNetwork(nbOfInputs, nbOfOutputs, sizeLayers);
            nn.Initialise();

            double[][] dataset = new double[sizeDataSet][];
            double[][] expected = new double[sizeDataSet][];

            for (int i = 0; i < sizeDataSet; i++)
            {
                dataset[i] = new double[nbOfInputs];
                dataset[i][0] = Rand.Next(2);
                dataset[i][1] = Rand.Next(2);

                expected[i] = new double[1];
                expected[i][0] = dataset[i][0] + dataset[i][1];
                if (expected[i][0] == 2)
                    expected[i][0] = 1;
            }

            var Trainer = new NNTrainer();

            Trainer.Train(nn, dataset, expected, 0.5, 15);
        }


    }
}
