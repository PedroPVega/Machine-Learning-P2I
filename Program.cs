// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");
Simulation sim1 = new Simulation();
int K = 3;
sim1.LoadData();
sim1.CreateRandomBarycenters(K);
sim1.InitializeKMeans();
for (int i = 0; i < 10; i++)
{
    Console.WriteLine();
    Console.WriteLine("Boucle numéro : {0}", i+1);
    sim1.KmeansIteration(K);
}

