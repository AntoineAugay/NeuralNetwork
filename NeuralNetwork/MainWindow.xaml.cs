using System;
using System.Windows;
using NeuralNetwork.Model;

namespace NeuralNetwork
{
    /// <summary>
    /// Logique d'interaction pour MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            NNManager Manager = new NNManager();
        }
    }
}
