using System;

namespace NandNN
{
	/// <summary>
	/// Learn the NAND function.
	/// </summary>
	class NandNN
	{
		const double learningRate = 0.8;
		const double desiredAccuracy = 0.001;
		const int hiddenNodeCount = 7;

		double[][] inputSequence;

		double[] inputNodes;
		double[][] inputWeights;
			
		double[] hiddenNodes;
		double[] hiddenWeights;

		double output;
		double[] expectedOutput;

		Random generator;

		[STAThread]
		static void Main(string[] args)
		{
			NandNN nandLearner = new NandNN();
			nandLearner.Run();
		}

		public NandNN()
		{
			string mode = "xor";


			switch(mode)
			{
				case "nand":
				{
					// NAND
					inputSequence = new double[][]
					{
						new double[] { 0, 0 },
						new double[] { 0, 1 },
						new double[] { 1, 0 },
						new double[] { 1, 1 }
					};
					expectedOutput = new double[] { 1, 1, 1, 0 };

				}; break;

				case "xor":
				{
					// XOR
					inputSequence = new double[][]
					{
						new double[] { 0, 0 },
						new double[] { 0, 1 },
						new double[] { 1, 0 },
						new double[] { 1, 1 }
					};
					expectedOutput = new double[] { 0, 1, 1, 0 };

				}; break;

				case "nandextra":
				{
					// NAND w/ extra (useless) input nodes
					inputSequence = new double[][]
					{
						new double[] { 0, 0, 0.20, 0.43 },
						new double[] { 0, 1, 0.98, 0.13 },
						new double[] { 1, 0, 0.50, 0.99 },
						new double[] { 1, 1, 0.43, 0.00 }
					};
					expectedOutput = new double[] { 1, 1, 1, 0 };

				}; break;
			}

			inputNodes = new double[inputSequence[0].Length + 1];
			inputWeights = new double[inputNodes.Length][];
			
			hiddenNodes = new double[hiddenNodeCount];
			hiddenWeights = new double[hiddenNodes.Length];

			output = 0;

			generator = new Random();
		}

		public void Run()
		{
			InitWeights();
			int trainIndex = 0;
			int iterationCount = 0;

			double[] errors =
			{
				Double.MaxValue,
				Double.MaxValue,
				Double.MaxValue,
				Double.MaxValue
			};

			do
			{
				trainIndex = generator.Next(4);
				FeedForward(inputSequence[trainIndex]);

				BackProp(expectedOutput[trainIndex]);
				iterationCount++;
				
				double currentError = Math.Abs(CalcError(expectedOutput[trainIndex], output));
				errors[trainIndex] = currentError;
			}
			while(errors[0] + errors[1] + errors[2] + errors[3] > desiredAccuracy);

			Console.WriteLine("Converged in " + iterationCount + " iterations.");
			for(int counter = 0; counter < 4; counter++)
				TestTraining(counter);
		}

		public void TestTraining(int which)
		{
			Console.WriteLine("Testing {0} {1}", inputSequence[which][0], inputSequence[which][1]);
 
			FeedForward(inputSequence[which]);
			Console.WriteLine("Expected: " + expectedOutput[which]);
			Console.WriteLine("Got: " + output);
	}

		private void InitWeights()
		{
			for(int counter = 0; counter < inputWeights.Length; counter++)
			{
				inputWeights[counter] = new double[hiddenNodes.Length];
				for(int outCounter = 0; outCounter < hiddenNodes.Length; outCounter++)
					inputWeights[counter][outCounter] = generator.NextDouble();
			}
			
			for(int counter = 0; counter < hiddenWeights.Length; counter++)
				hiddenWeights[counter] = generator.NextDouble();
		}

		private void FeedForward(double[] input)
		{
			// Prepare the input nodes
			inputNodes[0] = 1;
			for(int inputCounter = 1; inputCounter <= input.Length; inputCounter++)
				inputNodes[inputCounter] = input[inputCounter - 1];

			// Prepare the hidden nodes
			hiddenNodes[0] = 1;
			for(int counter = 1; counter < hiddenNodes.Length; counter++) hiddenNodes[counter] = 0;
			
			// Feed input to hidden
			for(int counter = 1; counter < hiddenNodes.Length; counter++)
			{
				for(int inputCounter = 0; inputCounter < inputNodes.Length; inputCounter++)
					hiddenNodes[counter] += inputNodes[inputCounter] * inputWeights[inputCounter][counter];
				hiddenNodes[counter] = Activation(hiddenNodes[counter]);
			}

			// Feed hidden to output
			output = 0;
			for(int counter = 0; counter < hiddenNodes.Length; counter++)
				output += hiddenNodes[counter] * hiddenWeights[counter];

			output = Activation(output);
		}

		private void BackProp(double expectedOutput)
		{
			double error = (expectedOutput - output);
			//Console.WriteLine("Current Error is: " + error);
			
			double backPropError = error * ActivationDerivative(output);
			//Console.WriteLine("Need to back-prop " + backPropError);

			double hiddenError = 0;

			// Update the output -> hidden weights
			for(int counter = 0; counter < hiddenNodes.Length; counter++)
			{
				hiddenError = ActivationDerivative(hiddenNodes[counter]) * backPropError * hiddenWeights[counter];
				hiddenWeights[counter] += (learningRate * backPropError * hiddenNodes[counter]);

				// Update the hidden -> input weights
				for(int inputCounter = 0; inputCounter < inputNodes.Length; inputCounter++)
					inputWeights[inputCounter][counter] += (learningRate * hiddenError * inputNodes[inputCounter]);
			}
		}

		private double Activation(double input)
		{
			return (1 / (1 + (Math.Exp(-input))));
		}

		private double ActivationDerivative(double input)
		{
			return (input * (1 - input));
		}

		private double CalcError(double expectedResult, double actualResult)
		{
			return (expectedResult - actualResult) * (expectedResult - actualResult);
			//return (expectedResult - actualResult);
		}
	}
}
