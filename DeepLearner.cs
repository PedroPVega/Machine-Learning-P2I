namespace Unsupervised_Learning;
public class DeepLearner : Simulation2
{
    static double Epsilon = 0.0000000001d;
    private List<double[,]> W {get; set;}
    private List<double[,]> dW {get; set;}
    private List<double[]> B {get; set;}
    private List<double[]> dB {get; set;}
    private double[][] A {get; set;}
    private double[][] Z {get; set;}
    public int L {get; set;}
    public int l {get; set;}

    public DeepLearner(int bigL, int littleL):base()
    {
        L = bigL;
        l = littleL;
        W = new List<double[,]>();
        dW = new List<double[,]>();
        B = new List<double[]>();
        dB = new List<double[]>();
        Z = new double[l][];
        A = new double[l][];
    }

    public void InitializeDeepLearning()
    {
        Console.WriteLine("Initializing matrixes and vectors");
        for (int i = 0; i < l; i++)
        {
            Z[i] = new double[L];
            A[i] = new double[L];
        }
        W.Add(new double[L,ImgSize]);
        dW.Add(new double[L,ImgSize]);
        B.Add(new double[L]);
        dB.Add(new double[L]);
        //FillRandomblyMatrix(W[0]);
        XavierInitializationMatrix(W[0]);
        InitializeBiasColumn(B[0]);
        for (int i = 1; i < l; i++)
        {
            W.Add(new double[L,L]);
            dW.Add(new double[L,L]);
            
            B.Add(new double[L]);
            dB.Add(new double[L]);

            //FillRandomblyMatrix(W[i]);
            XavierInitializationMatrix(W[i]);
            InitializeBiasColumn(B[i]);
        }
        W.Add(new double[10,L]);
        dW.Add(new double[10,L]);
        
        B.Add(new double[10]);
        dB.Add(new double[10]);

        //FillRandomblyMatrix(W[l]);
        XavierInitializationMatrix(W[l]);
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
            Z[0][h] += + bi[h];
            A[0][h] = Sigmoid(Z[0][h]);
            /*
            if (A[0][h] is double.NaN)
            {
                Console.WriteLine("activation is too small");
            }
            if (A[0][h] == double.PositiveInfinity)
            {
                Console.WriteLine("activation is too high");
            }
            */
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
                    Z[h][i] += wi[j,h] * A[h][i-1];
                }
                Z[h][i] += bi[h];
                A[h][i] = Sigmoid(Z[h][i] );
                
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
                if (wi[h,j] == 0.0)
                {
                    //Console.WriteLine("ERREUR");
                }
            }
            prediction[h] += bi[h];
        }
        //Console.WriteLine("longueur de la prediction : {0}", prediction.Length);
        double min = prediction.Min();
        double max = prediction.Max();
        //Console.WriteLine("minimum of z2 : {0} ; maximum of z2 : {1}", min, max);
        //Console.WriteLine("output layer before softmax");
        //PrintArray(prediction);
        prediction = SoftMax(prediction);
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
    public double Backwards(Number Num, double[] y)
    {
        // WEIGHTS FOR 1 NUMBER 1 LAYER OF 1 NEURON
        double[] vect_label = new double[y.Length];
        vect_label[Num.Label] = 1d;
        double loss = CrossEntropyLoss(vect_label,y);
        double[] delta2 = new double[y.Length];
        double[] delta1 = new double[L];

        // OUTPUT LAYER
        for (int i = 0; i < delta2.Length; i++)
            delta2[i] = y[i] - vect_label[i];

        for (int i = 0; i < y.Length; i++)
        {
            dB[l][i] += delta2[i];
            for (int h = 0; h < L; h++)
            {
                //Console.WriteLine("{0} x {1}",dW[l].GetLength(0),dW[l].GetLength(1));
                //Console.WriteLine("h = {0} ; i = {1}", h,i);
                //Console.WriteLine("{0}",dW[l][i,h]);
                dW[l][i,h] += delta2[i] * A[l-1][h];
            }
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
            delta1[h] *= d_dx_Sigmoid(Z[0][h]);
        }

        for (int h = 0; h < L; h++)
        {
            dB[0][h] += delta1[h];
            for (int i = 0; i < ImgSize; i++)
            {
                dW[0][h,i] += delta1[h] * Num.Pixels[i];
            }
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
    public double[] SoftMax(double[] z)
    {
        double[] a = new double[z.Length];
        double sum = 0;
        for (int i = 0; i < z.Length; i++)
        {
            sum += Math.Exp(z[i]);
        }
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = Math.Exp(z[i]) / sum;
        }
        return a;
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
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = Math.Exp(z[i]) / temp;
        }
        return a;
    }
    public double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
    public double d_dx_Sigmoid(double x)
    {
        double term = Math.Exp(-x);
        return term / Math.Pow(term + 1,2);
    }
    public void Train(int nbEpochs, int batchSize, double learningRate)
    {
        /*
        Number num = Numbers[0];
        double[] pred = Forward(num.Pixels);
        Backwards(num, pred);
        */
        List<List<Number>> Batches = GetBatches(nbEpochs,batchSize);
        List<Number> batch0 = Batches[0];
        double[] y = new double[10];
        int counter = 1;
        double lr = learningRate;
        double lossOverTime = 0;
        double acc = 0;
        foreach (List<Number> batch in Batches)
        {
            
            foreach (Number num in batch0)
            {
                y = Forward(NormalizeVector(num.Pixels));
                if (Math.Round(y.Sum()) != 1)
                {
                    Console.WriteLine("ERROR {0}",y.Sum());
                    PrintArray(y);
                }
                if (GetPrediction(y) == num.Label) { acc += 1; }
                lossOverTime += Backwards(num,y);
                //Console.WriteLine("The number was a {0}",num.Label);
            }
            
            
            //ShowWeights();
            //PrintArray(y);
            lossOverTime /= batchSize;
            acc = acc / batchSize * 100;
            SetAverageGradient(batchSize);
            UpdateWAndB(lr);
            ResetDWAndDB();
            Console.WriteLine("Epoch number {0} finished training ! Average loss : {1} ; Accuracy : {2} %", counter++, lossOverTime, Math.Round(acc,4));
            lr *= 0.95;
            lossOverTime = 0;
            acc = 0;
            
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
                Console.WriteLine("interessant");
                return pred;
            }
        }
        return pred;
    }

    public void SetAverageGradient(int batchSize)
    {
        foreach (double[,] wi in dW)
        {
            DivideMatrixByScalar(wi,batchSize);
        }
        foreach (double[] bi in dB)
        {
            DivideArrayByScalar(bi,batchSize);
        }
    }

    public void DivideArrayByScalar(double[] ary, int q)
    {
        for (int j = 0; j < ary.Length; j++)
        {
            ary[j] /= q;
        }
    }

    public void DivideMatrixByScalar(double[,] mat, int q)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i,j] /= q;
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
                mat[i,j] = ((rdn.Next(10)-5)/10) * fact;
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
    public List<List<Number>> GetBatches(int nbEpochs, int batchSize)
    {
        List<List<Number>> Batches = new List<List<Number>>();
        int upEnd = Numbers.Count / batchSize;
        int i;
        int[] alreadyChosen = new int[nbEpochs];
        for (int j = 0; j < nbEpochs; j++)
        {
            i = (Int32)rdn.NextInt64(upEnd);
            if (!alreadyChosen.Contains(i))
            {
                alreadyChosen[j] = i;
                Batches.Add(Numbers.GetRange(i*batchSize, batchSize));
            }
            else
            {
                j--;
            }
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
            vector[i] = array[i]/255d;
        }
        return vector;
    }
}