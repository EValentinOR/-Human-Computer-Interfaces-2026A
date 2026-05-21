#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

class MultipleLinearRegression
{
private:
    vector<vector<double>> X;
    vector<double> y;
    vector<double> betas;
    int N, p;

    vector<vector<double>> transpose(const vector<vector<double>>& mat)
    {
        int rows = mat.size(), cols = mat[0].size();
        vector<vector<double>> res(cols, vector<double>(rows));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                res[j][i] = mat[i][j];
        return res;
    }

    vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B)
    {
        int rA = A.size(), cA = A[0].size(), cB = B[0].size();
        vector<vector<double>> res(rA, vector<double>(cB, 0.0));
        for (int i = 0; i < rA; i++)
            for (int j = 0; j < cB; j++)
                for (int k = 0; k < cA; k++)
                    res[i][j] += A[i][k] * B[k][j];
        return res;
    }

    vector<double> multiplyMV(const vector<vector<double>>& A, const vector<double>& b)
    {
        int rows = A.size(), cols = A[0].size();
        vector<double> res(rows, 0.0);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                res[i] += A[i][j] * b[j];
        return res;
    }

    vector<double> solveLinearSystem(vector<vector<double>> A, vector<double> b)
    {
        int n = A.size();
        for (int i = 0; i < n; i++)
        {
            for (int k = i+1; k < n; k++)
            {
                double factor = A[k][i] / A[i][i];
                for (int j = i; j < n; j++)
                    A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        vector<double> sol(n);
        for (int i = n-1; i >= 0; i--)
        {
            sol[i] = b[i];
            for (int j = i+1; j < n; j++)
                sol[i] -= A[i][j] * sol[j];
            sol[i] /= A[i][i];
        }
        return sol;
    }

public:
    MultipleLinearRegression(const vector<vector<double>>& X_in, const vector<double>& y_in)
        : X(X_in), y(y_in)
    {
        N = y.size();
        p = X[0].size() + 1;
        betas.resize(p);
    }

    void fit()
    {
        vector<vector<double>> Xd(N, vector<double>(p, 1.0));
        for (int i = 0; i < N; i++)
            for (int j = 1; j < p; j++)
                Xd[i][j] = X[i][j-1];

        auto Xt = transpose(Xd);
        auto XtX = multiply(Xt, Xd);
        auto XtY = multiplyMV(Xt, y);
        betas = solveLinearSystem(XtX, XtY);
    }

    vector<double> getBetas()
    {
        return betas;
    }
    double predict(double x1, double x2) const
    {
        return betas[0] + betas[1] * x1 + betas[2] * x2;
    };
};


int main()
{
    vector<vector<double>> X =
    {
        {41.9,29.1}, {43.4,29.3}, {43.9,29.5}, {44.5,29.7},
        {47.3,29.9}, {47.5,30.3}, {47.9,30.5}, {50.2,30.7},
        {52.8,30.8}, {53.2,30.9}, {56.7,31.5}, {57.0,31.7},
        {63.5,31.9}, {65.3,32.0}, {71.1,32.1}, {77.0,32.5},
        {77.8,32.9}
    };
    vector<double> y = {251.3,251.3,248.3,267.5,273.0,276.5,270.3,274.9,
                        285.0,290.0,297.0,302.5,304.5,309.3,321.7,330.7,349.0
                       };
    MultipleLinearRegression model(X, y);
    model.fit();
    auto betas = model.getBetas();

    cout << fixed << setprecision(2);
    cout << "          Comparativa"<< "\n-------------------------------------"<< "\n Parametro | Valores Calculados | Valores Predichos " << endl;
    cout << "    B0     |     " << betas[0] << "      |     -153.51"<<"\n    B1     |       " << betas[1] << "       |      1.24" << "\n    B2     |      "<< betas[2] << "       |      12.08" << endl;

    cout << "\nEcuacion de regresion\n";
    cout << "Yield = " << betas[0] << " + " << betas[1] << " * Factor1 (X1) + "
         << betas[2] << " * Factor2 (X2) \n";

    cout << "\n            Simulacion de nuevos experimentos\n";
    cout << " Experimento |       x1       |        x2       |  Yield predicho\n";
    cout << "-------------------------------------------------------------\n";

    vector<pair<double, double>> nuevosExperimentos =
    {
        {45.7, 30.2}, {52.9, 31.3}, {64.5, 32.4}, {73.1, 32.7}, {80.0, 33.0}
    };

    for (size_t i = 0; i < nuevosExperimentos.size(); ++i)
    {
        double x1 = nuevosExperimentos[i].first;
        double x2 = nuevosExperimentos[i].second;
        double y_pred = model.predict(x1, x2);
        cout << "     " << i+1 << "      |     " << setw(8) << x1 << "    |     "
             << setw(8) << x2 << "    |     " << setw(8) << y_pred << "\n";
    }

    return 0;
}
