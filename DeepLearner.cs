namespace Unsupervised_Learning;
public class DeepLearner : Simulation2
{
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
        Z = new double[L][];
        A = new double[L][];
    }

    public void InitializeDeepLearning()
    {
        Console.WriteLine("Initializing matrixes and vectors");
        for (int i = 0; i < L; i++)
        {
            Z[i] = new double[l];
            A[i] = new double[l];
        }
        //W.Add(new double[ImgSize,L]);
        W.Add(new double[ImgSize,L]);
        dW.Add(new double[ImgSize,L]);
        B.Add(new double[L]);
        dB.Add(new double[L]);
        FillRandomblyVector(B[0]);
        for (int i = 1; i < l; i++)
        {
            W.Add(new double[L,L]);
            dW.Add(new double[L,L]);
            
            B.Add(new double[L]);
            dB.Add(new double[L]);

            FillRandomblyMatrix(W[i]);
            FillRandomblyVector(B[i]);
        }
        W.Add(new double[L,10]);
        dW.Add(new double[L,10]);
        
        B.Add(new double[10]);
        dB.Add(new double[10]);

        FillRandomblyMatrix(W[l]);
        FillRandomblyVector(B[l]);
        Console.Clear();
        Console.WriteLine("Neural Network Initialized !!");
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
        return Forward(num.Pixels);
    }
    public double[] Forward(int[] vector)
    {
        // DECLARING VARIABLES
        double[,] wi; // weight i
        double[] bi; // biais i
        double[] prediction = new double[10];

        // FIRST COLUMN
        wi = W[0];
        bi = B[0];
        //Console.WriteLine("Starting layer 0");
        for (int h = 0; h < L; h++)
        {
            for (int j = 0; j < ImgSize; j++)
            {
                Z[h][0] += wi[j,h] * vector[j];
            }
            //PrintArray(bi);
            //PrintArray(Z[h]);
            Z[h][0] += + bi[h];
            A[h][0] = Sigmoid(Z[h][0]);
        }
        //A[0] = SoftMax(Z[0]);

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
            //A[l] = SoftMax(Z[l]);
        }

        // LAST COLUMN
        wi = W[W.Count-1];
        bi = B[B.Count-1];
        for (int h = 0; h < prediction.Length; h++)
        {
            for (int j = 0; j < L; j++)
            {
                prediction[h] += wi[j,h] * A[j][A[j].Length-1];
            }
            prediction[h] += bi[h];
            //prediction[h] = Sigmoid(prediction[h]);
        }
        A[A.GetLength(0)-1] = SoftMax(Z[Z.GetLength(0)-1]);
        Console.WriteLine("Forward Pass finished");
        return prediction;
    }

    public void Backwards(Number Num, double[] y)
    {
        // WEIGHTS FOR 1 NUMBER 1 LAYER OF 1 NEURON
        double[] vect_label = new double[10];
        vect_label[Num.Label] = 1d;
        double loss = CrossEntropyLoss(Num.Label,y);
        double[] delta2 = new double[y.Length];
        double[] delta1 = new double[L];

        // OUTER LAYER
        for (int i = 0; i < delta2.Length; i++)
                delta2[i] = y[i] - vect_label[i];

        for (int h = 0; h < L; h++)
        {
            for (int i = 0; i < y.Length; i++)
            {
                //Console.WriteLine("{0} x {1}",dW[1].GetLength(0),dW[1].GetLength(1));
                //Console.WriteLine("{0}",delta2[i]);
                //Console.WriteLine("{0}",A[h][i]);
                dW[1][h,i] = delta2[i] * A[h][0];
                dB[1][i] = delta2[i];
            }
        }
        
        // INNER LAYERS

        // FIRST LAYER
        for (int h = 0; h < L; h++)
        {
            for (int i = 0; i < delta2.Length; i++)
                delta1[h] += delta2[i] * W[1][h,i];
            delta1[h] *= d_dx_Sigmoid(Z[h][0]);
        }

        for (int h = 0; h < L; h++)
        {
            for (int i = 0; i < ImgSize; i++)
            {
                dW[0][i,h] = delta1[h] * Num.Pixels[i];
                dB[0][h] = delta1[h];
            }
        }
        
        Console.WriteLine("Backwards Pass Completed");
    }

    public double CrossEntropyLoss(int label, double[] prediction)
    {
        double L = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            L += label*Math.Log(prediction[i]);
        }

        return - L;
    }

    public double[] SoftMax(double[] z)
    {
        double[] a = new double[z.Length];
        double temp = 0;
        for (int i = 0; i < z.Length; i++)
        {
            temp += Math.Exp(z[i]);
        }
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = z[i] / temp;
        }
        return a;
    }

    public void Train()
    {
        Number num = Numbers[0];
        /*
        double[] vect = new double[ImgSize];
        for (int i = 0; i < ImgSize; i++)
        {
            vect[i] = num.Pixels[i]/255d;        
        }
        */
        double[] pred = Forward(num.Pixels);
        Backwards(num, pred);
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
}