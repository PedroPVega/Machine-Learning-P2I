namespace Unsupervised_Learning;
public class DeepLearner : Simulation2
{
    static double Epsilon = 0.0000000001d;
    private List<double[,]> W {get; set;}
    private List<double[,]> dW {get; set;}
    private List<double[]> B {get; set;}
    private List<double[]> dB {get; set;}
    public List<(double[] x, double[] y)> XTrain_YTrain {get; set;}
    private double[][] A {get; set;}
    private double[][] Z {get; set;}
    public double LearningRate{get; set;}   
    public int L {get; set;}
    public int l {get; set;}

    public DeepLearner(int bigL, int littleL, double Lr):base()
    {
        L = bigL;
        l = littleL;
        LearningRate = Lr;
        W = new List<double[,]>();
        dW = new List<double[,]>();
        B = new List<double[]>();
        dB = new List<double[]>();
        XTrain_YTrain = new List<(double[] x, double[] y)>();
        Z = new double[l][];
        A = new double[l][];
        Console.WriteLine("Loading data");
        LoadData();
        Console.Clear();
    }

    public void InitializeDeepLearning()
    {
        Console.WriteLine("Initializing matrixes and vectors");
        for (int i = 0; i < l-1; i++)
        {
            Z[i] = new double[L];
            A[i] = new double[L];
        }
        A[l-1] = new double[10];
        Z[l-1] = new double[10];
        W.Add(new double[L,ImgSize]);
        dW.Add(new double[L,ImgSize]);
        B.Add(new double[L]);
        dB.Add(new double[L]);
        HeInitializationMatrix(W[0]);
        InitializeBiasColumn(B[0]);
        for (int i = 1; i < l; i++)
        {
            W.Add(new double[L,L]);
            dW.Add(new double[L,L]);
            
            B.Add(new double[L]);
            dB.Add(new double[L]);

            HeInitializationMatrix(W[i]);
            InitializeBiasColumn(B[i]);
        }
        W.Add(new double[10,L]);
        dW.Add(new double[10,L]);
        
        B.Add(new double[10]);
        dB.Add(new double[10]);

        //FillRandomblyMatrix(W[l]);
        HeInitializationMatrix(W[l]);
        InitializeBiasColumn(B[l]);
        Console.Clear();
        Console.WriteLine("Neural Network Initialized !!");
        //FillWeights();
    }
    public double[] ForwardRandom()
    {
        int idx = (int)rdn.NextInt64(Numbers.Count);
        //double[] vect = new double[ImgSize];
        Number num = Numbers[idx];
        /*
        for (int i = 0; i < ImgSize; i++)
        {
            vect[i] = num.Pixels[i]/255d;        
        }
        */
        return Forward(NormalizeVector(num.Pixels));
    }
    public double[] Forward(double[] vector)
    {
        // DECLARING VARIABLES
        double[,] wi; // weight i
        double[] bi; // biais i
        double[] prediction = new double[10];
        //Console.WriteLine("There are {0} weight matrixes",W.Count);
        //Console.WriteLine("There are {0} biais columns",B.Count);

        // FIRST COLUMN
        wi = W[0];
        bi = B[0];
        for (int h = 0; h < L; h++)
        {
            for (int j = 0; j < ImgSize; j++)
            {
                Z[0][h] += wi[h,j] * vector[j];
            }
            Z[0][h] += bi[h];
            A[0][h] = RelU(Z[0][h]);
            
            if (A[0][h] is double.NaN)
            {
                Console.WriteLine("activation is too small");
            }
            if (A[0][h] == double.PositiveInfinity)
            {
                Console.WriteLine("activation is too high");
            }
            
            //PrintArray()
        }

        // SECOND COLUMN THROUGH l COLUMN
        for (int i = 1; i < l; i++) 
        {
            // FOREACH HIDDEN LAYER
            wi = W[i];
            bi = B[i];
            for (int h = 0; h < L; h++)
            {
                for (int j = 0; j < L; j++)
                {
                    Z[i][h] += wi[j,h] * A[i-1][h];
                }
                Z[i][h] += bi[h];
                A[i][h] = RelU(Z[i][h] );
                
            }
        }

        // LAST COLUMN
        wi = W[W.Count-1];
        bi = B[B.Count-1];
        for (int h = 0; h < prediction.Length; h++)
        {
            for (int j = 0; j < L; j++)
            {
                prediction[h] += wi[h,j] * A[l-1][j];
            }
            prediction[h] += bi[h];
        }
        //Console.WriteLine("longueur de la prediction : {0}", prediction.Length);
        //double min = prediction.Min();
        //double max = prediction.Max();
        //Console.WriteLine("minimum of z2 : {0} ; maximum of z2 : {1}", min, max);
        //Console.WriteLine("output layer before softmax");
        //PrintArray(prediction);
        SoftMax2(prediction);
        //Console.WriteLine("output layer after softmax");
        //PrintArray(prediction);
        if (Math.Round(prediction.Sum()) != 1)
        {
            Console.WriteLine("ERROR {0}",prediction.Sum());
            PrintArray(prediction);
        }
        //Console.WriteLine("Forward Pass finished");
        //Console.WriteLine("longueur de la prediction : {0}", prediction.Length);
        return prediction;
    }

    public double[] ForwardChat(double[] x)
    {
        A[0] = x;
        double z;
        for (int i = 1; i < l; i++) // Hidden layers
        {
            for (int h = 0; h < L; h++)
            {
                z = 0;
                for (int j = 0; j < L; j++)
                    z += W[i][h, j] * A[i - 1][j];
                z += B[i][h];
                Z[i][h] = z;
                A[i][h] = RelU(z);
            }
        }

        // Output layer
        for (int o = 0; o < 10; o++)
        {
            z = 0;
            for (int h = 0; h < L; h++)
                z += W[l][o, h] * A[l - 1][h];
            z += B[l][o];
            Z[l-1][o] = z;
        }

        A[l-1] = SoftMax2(Z[l-1]); // stable softmax
        return A[l-1];
    }
    public void BackwardChat(double[] x, double[] y)
    {
        ForwardChat(x);

        // Gradients
        double[][] dZ = new double[l + 1][];
        double[] predicted = A[l-1];

        // Output layer delta
        dZ[l] = new double[10];
        for (int i = 0; i < 10; i++)
            dZ[l][i] = predicted[i] - y[i];

        // Hidden layers delta
        for (int i = l - 1; i > 0; i--)
        {
            dZ[i] = new double[L];
            for (int h = 0; h < L; h++)
            {
                double sum = 0;
                for (int k = 0; k < W[i + 1].GetLength(0); k++)
                    sum += dZ[i + 1][k] * W[i + 1][k, h];
                dZ[i][h] = sum * dRelU(Z[i][h]);
            }
        }

        // Gradient updates for weights and biases
        for (int i = l; i > 0; i--)
        {
            int rows = W[i].GetLength(0);  // output size
            int cols = W[i].GetLength(1);  // input size

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    double grad = dZ[i][r] * A[i - 1][r];
                    W[i][r, c] -= LearningRate * grad;
                }
                B[i][r] -= LearningRate * dZ[i][r];
            }
        }
    }
    public double Backwards(Number Num, double[] y)
    {
        // WEIGHTS FOR 1 NUMBER 1 LAYER OF 1 NEURON
        double[] vect_label = new double[y.Length];
        vect_label[Num.Label] = 1.0;
        double loss = CrossEntropyLoss(vect_label,y);
        double[] delta2 = new double[y.Length];
        double[] delta1 = new double[L];

        // OUTPUT LAYER
        for (int i = 0; i < delta2.Length; i++)
            delta2[i] = y[i] - vect_label[i];

        for (int i = 0; i < y.Length; i++)
        {
            for (int h = 0; h < L; h++)
            {
                //Console.WriteLine("{0} x {1}",dW[l].GetLength(0),dW[l].GetLength(1));
                //Console.WriteLine("h = {0} ; i = {1}", h,i);
                //Console.WriteLine("{0}",dW[l][i,h]);
                //dW[l][i,h] += delta2[i] * A[l-1][h];
                dW[l][i,h] += delta2[i]*A[l-1][h];
            }
            dB[l][i] += delta2[i];
        }
        
        // HIDDEN LAYERS

        // INPUT LAYER
        for (int h = 0; h < W[1].GetLength(1); h++)
        {
            for (int i = 0; i < W[1].GetLength(0); i++)
            {
                //Console.WriteLine("{0}",)
                delta1[h] += delta2[i] * W[1][i,h]; // Ce n'est plus delta2, à changer plus tard
            }
            delta1[h] *= dRelU(Z[0][h]);
        }

        for (int h = 0; h < L; h++)
        {
            for (int i = 0; i < ImgSize; i++)
            {
                dW[0][h,i] += delta1[h] * Num.Pixels[i];
            }
            dB[0][h] += delta1[h];
        }
        //Console.WriteLine("Backwards Pass Completed");
        return loss;
    }
    public double CrossEntropyLoss(double[] vect_label, double[] prediction)
    {
        double L = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            L += vect_label[i]*Math.Log(prediction[i] + Epsilon);
        }
        //Console.WriteLine("Loss for this number : {0}",Math.Round(-L,4));
        return - L;
    }
    public void SoftMax(double[] z)
    {
        //double[] a = new double[z.Length];
        double sum = 0;
        double check = 0;
        for (int i = 0; i < z.Length; i++)
        {
            sum += Math.Exp(z[i]);
        }
        for (int i = 0; i < z.Length; i++)
        {
            z[i] = Math.Exp(z[i]) / sum;
            check += z[i];
        }
        Console.WriteLine("sum of softmax : {0}",check);
    }
    public double[] SoftMax2(double[] z)
    {
        double mx = z.Max();
        double[] a = new double[z.Length];
        double temp = 0;
        for (int i = 0; i < z.Length; i++)
        {
            z[i] -= mx;
            temp += Math.Exp(z[i]);
        }
        for (int i = 0; i < z.Length; i++)
        {
            a[i] = Math.Exp(z[i]) / temp;
        }
        return a;
    }
    public double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    public double Tanh(double x)
    {
        double pos = Math.Exp(x);
        double neg = Math.Exp(-x);
        return (pos - neg) / (pos + neg);
    }
    public double RelU(double x)
    {
        if (x > 0)
            return x;
        else
            return 0;
    }
    public double d_dx_Sigmoid(double x)
    {
        double term = Math.Exp(-x);
        return term / Math.Pow(term + 1,2);
    }
    public double dTanh(double x)
    {
        double squared = Math.Pow(Tanh(x),2);
        return 1 - squared;
    }
    public double dRelU(double x)
    {
        if (x > 0)
            return 1;
        else
            return 0;
    }
    public void Train(int nbEpochs, int batchSize, double learningRate)
    {
        /*
        Number num = Numbers[0];
        double[] pred = Forward(num.Pixels);
        Backwards(num, pred);
        */
        Console.WriteLine("Retreiving batches");
        List<List<Number>> Batches = GetBatches(batchSize);
        //List<Number> batch0 = Batches[0];
        double[] y = new double[10];
        int counter = 1;
        int total =0;
        double lr = learningRate;
        double lossOverTime = 0;
        double acc = 0;
        for (int e = 0; e < nbEpochs; e++)
        {
            Console.WriteLine("Epoch {0} / {1}",e,nbEpochs);
            foreach (List<Number> batch in Batches)
            {
                foreach (Number num in batch)
                {
                    y = Forward(NormalizeVector(num.Pixels));
                    if (Math.Round(y.Sum()) != 1)
                    {
                        Console.WriteLine("ERROR {0}",y.Sum());
                        PrintArray(y);
                    }
                    if (GetPrediction(y) == num.Label) { acc += 1; }
                    total += 1;
                    lossOverTime += Backwards(num,y);
                    
                    //Console.WriteLine("The number was a {0}",num.Label);
                    //PrintArray(y);
                }
                //ShowWeights();
                SetAverageGradient(batch.Count);
                UpdateWAndB(lr);
                ResetDWAndDB();
            }
            
            lossOverTime /= total;
            acc = acc / total * 100;
            Console.WriteLine("Epoch number {0} finished training ! Average loss : {1} ; Accuracy : {2} %", counter++, lossOverTime, Math.Round(acc,4));
            //lr *= 0.95;
            lossOverTime = 0;            
            acc = 0;
            total = 0;
        }
    }
    public void Fit(List<(double[] x, double[] y)> dataset, int epochs, int batchSize)
    {
        int n = dataset.Count;
        int nbGuessed = 0;
        int total = 0;
        double acc;
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Shuffle(dataset); // Shuffle before each epoch
            double totalLoss = 0;

            for (int i = 0; i < n; i += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - i);
                var batch = dataset.GetRange(i, actualBatchSize);

                for (int j = 0; j < actualBatchSize; j++)
                {
                    var (x, y) = batch[j];
                    double[] output = ForwardChat(x);
                    totalLoss += CrossEntropyLoss(output, y);
                    BackwardChat(x, y);
                    if (GetPrediction(output) == GetPrediction(y))
                    {
                        nbGuessed += 1;
                    }
                    total += 1;
                }
            }
            acc = nbGuessed/total * 100;
            Console.WriteLine($"Epoch {epoch}/{epochs} - Avg Loss: {totalLoss / n:F4} - Avg Accuracy: {nbGuessed}/{total}");
            acc = 0;
            total = 0;
            nbGuessed = 0;
        }
    }
    public void Shuffle(List<(double[] x, double[] y)> dataset)
    {
        Random rng = new Random();
        int n = dataset.Count;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            var temp = dataset[k];
            dataset[k] = dataset[n];
            dataset[n] = temp;
        }
    }
    public void UpdateWAndB(double lR)
    {
        for(int t = 0; t < W.Count; t++)
        {
            for (int i = 0; i < W[t].GetLength(0); i++)
            {
                B[t][i] -= lR * dB[t][i];
                for (int j = 0; j < W[t].GetLength(1); j++)
                {
                    //Console.WriteLine("valeur avant : {0}",W[t][i,j]);
                    //Console.WriteLine("gradient : {0}",dW[t][i,j]);
                    W[t][i,j] -= lR * dW[t][i,j];
                    //Console.WriteLine("valeur apres : {0}",W[t][i,j]);
                }
            }
        }
    }

    public void ResetDWAndDB()
    {
        for(int t = 0; t < W.Count; t++)
        {
            for (int i = 0; i < W[t].GetLength(0); i++)
            {
                dB[t][i] = 0;
                for (int j = 0; j < W[t].GetLength(1); j++)
                {
                    dW[t][i,j] = 0;
                }
            }
        }
    }

    public int GetPrediction(double[] y)
    {
        double temp = double.MinValue;
        int pred = -1;
        for (int i = 0; i < y.Length; i++)
        {
            if (y[i] > temp)
            {
                pred = i;
                temp = y[i];
            }
            if (temp > 0.5)
            {
                //Console.WriteLine("interessant");
                return pred;
            }
        }
        return pred;
    }

    public void SetAverageGradient(int batchSize)
    {
        foreach (double[,] dwi in dW)
        {
            DivideMatrixByScalar(dwi,batchSize);
        }
        foreach (double[] dbi in dB)
        {
            DivideArrayByScalar(dbi,batchSize);
        }
    }

    public void DivideArrayByScalar(double[] ary, int q)
    {
        for (int j = 0; j < ary.Length; j++)
        {
            ary[j] = ary[j] / (double)q;
        }
    }

    public void DivideMatrixByScalar(double[,] mat, int q)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i,j] = mat[i,j] / (double)q;
            }
        }
    }
    public void FillRandomblyMatrix(double[,] mat)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                //mat[i,j] = rdn.NextDouble();
                mat[i,j] = 0.3;
            }
        }
    }
    public void XavierInitializationMatrix(double[,] mat)
    {
        /*
        Xavier (or Glorot) Initialization is a method used to initialize the weights of a neural network
        to prevent vanishing or exploding gradients during training.
        It ensures that the variance ofactivations remains consistent across layers.
        PARAMETERS : 
        - mat : weight matrix to fill
        */
        double fact = Math.Sqrt(1/ mat.GetLength(1));
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i,j] = rdn.NextDouble() * (2/784);
            }
        }
    }

    public void HeInitializationMatrix(double[,] mat)
    {
        /*
        Kaiming Initialization, or He Initialization, is an initialization method for neural networks
        that takes into account the non-linearity of activation functions, such as ReLU activations.
        A proper initialization method should avoid reducing or magnifying
        the magnitudes of input signals exponentially.
        PARAMETERS : 
        - mat : weight matrix to fill
        */
        double stdDev = 2 / mat.GetLength(1);
        double u1 = 1.0-rdn.NextDouble(); //uniform(0,1] random doubles
        double u2 = 1.0-rdn.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)

        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i,j] = stdDev * randStdNormal;
            }
        }
    }
    public void InitializeBiasColumn(double[] vect)
    {
        for (int i = 0; i < vect.Length; i++)
        {
            vect[i] = 0.01;
        }
    }

    public void FillWeights()
    {
        foreach (double[,] w in W)
        {
            for (int i = 0; i < w.GetLength(0); i++)
            {
                for (int j = 0; j < w.GetLength(1); j++)
                {
                    w[i,j] = 0.3d;
                }
            }
        }
    }
    public void ShowWeights()
    {
        foreach (double[,] w in dW)
        {
            int rows = w.GetLength(0);
            int cols = w.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write(w[i, j] + "\t"); // Print with tab spacing
                }
                Console.WriteLine("///////////////////////////////"); // Move to next line
            }
        }
    }
    public List<List<Number>> GetBatches(int batchSize)
    {
        List<List<Number>> Batches = new List<List<Number>>();
        int upEnd = Numbers.Count / batchSize;
        int i;
        //int[] alreadyChosen = new int[upEnd];
        for (int j = 0; j < upEnd; j++)
        {
            //i = (Int32)rdn.NextInt64(upEnd);
            //if (!alreadyChosen.Contains(i))
            //{
                //alreadyChosen[j] = i;
                Batches.Add(Numbers.GetRange(j*batchSize, batchSize));
            //}
            //else
            //{
                //j--;
            //}
        }
        return Batches;
    }
    public void PrintArray(double[] jaggedArray)
    {
        
        Console.Write("Array : ");
        foreach (double num in jaggedArray)
        {
            Console.Write(Math.Round(num,2) + " ");
        }
        Console.WriteLine();
        
    }
    public double[] NormalizeVector(int[] array)
    {
        double[] vector = new double[array.Length];
        for (int i = 0; i < array.Length; i++)
        {
            //vector[i] = (array[i] - 127.5d) / 127.5d;
            vector[i] = array[i] / 255d;
        }
        return vector;
    }
    public double ArrayMultiplication(double[] Activation, double [] Weights)
    {
        if (Activation.Length != Weights.Length)
        {
            Console.WriteLine("The calc is not good");
            return 0;
        }
        double result = 0;
        for (int i = 0; i < Activation.Length; i++)
        {
            result += Activation[i]*Weights[i];
        }
        return result;
    }

    public override void LoadData()
    {
        string[]? values;
        int[] xtrain = new int[784];
        string filePath = "../mnist_train.csv";
        using (StreamReader reader = new StreamReader(filePath))
        {
            string? line = reader.ReadLine(); // Ignore first line, titles
            for (int i = 0; i < 1000; i++) // Load first 1000 numbers
            {
                double[] ytrain = new double[10];
                line = reader.ReadLine();  // Read line
                if (line != null)
                {
                    values = line.Split(','); // Split by comma
                    ytrain[Convert.ToInt32(values[0])] = 1;
                    for (int t = 1; t < values.Length; t++) 
                        xtrain[t-1] = Convert.ToInt32(values[t]);     
                }
                XTrain_YTrain.Add((NormalizeVector(xtrain),ytrain));
            } 
        }
    }
}