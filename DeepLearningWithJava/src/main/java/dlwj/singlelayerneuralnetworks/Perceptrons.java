package dlwj.singlelayerneuralnetworks;

import static dlwj.util.ActivationFunction.step;

import java.util.Random;

import dlwj.util.GaussianDistribution;

public class Perceptrons {

	public int nIn;
	public double[] w;

	public Perceptrons(int nIn) {
		this.nIn = nIn;
		w = new double[nIn];		// 変数wにdouble型の配列オブジェクトを生成
	}

	public int train(double[] x, int t, double learningRate) {
		int classified = 0;
		double c = 0;

		// データが正しく分類されているかチェック。誤分類されているときのみ、勾配降下法を適用
		for (int i=0; i<nIn; i++) {
			c += w[i] * x[i] * t;
		}

		if (c > 0) {
			classified = 1;
		} else {
			for (int i=0; i<nIn; i++) {
				w[i] += learningRate * x[i] * t;
			}
		}

		return classified;
	}

	public int predict(double[] x) {
		double preActivation = 0.;

		for (int i=0; i<nIn; i++) {
			preActivation += w[i] * x[i];
		}

		return step(preActivation);
	}


	public static void main(String[] args) {
		final Random rng = new Random(1234);

		final int train_N = 1000;						// トレーニングデータ数
		final int test_N = 200;							// テストデータ数
		final int nIn = 2;								// 入力層のユニット数（ニューロン数）

		double[][] train_X = new double[train_N][nIn];	// トレーニングデータ
		int [] train_T = new int[train_N];				// トレーニングデータのラベル

		double[][] test_X = new double[test_N][nIn];		// テストデータ
		int[] test_T = new int[test_N];					// テストデータのラベル
		int[] predicted_T = new int[test_N];				// モデルの予測結果

		final int epochs = 2000;							// トレーニング数の上限
		final double learningRate = 1.;					// 学習率はパーセプトロンでは1でよい

		GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0,  1.0,  rng);

	       // data set in class 1
        for (int i = 0; i < train_N/2 - 1; i++) {
            train_X[i][0] = g1.random();
            train_X[i][1] = g2.random();
            train_T[i] = 1;
        }
        for (int i = 0; i < test_N/2 - 1; i++) {
            test_X[i][0] = g1.random();
            test_X[i][1] = g2.random();
            test_T[i] = 1;
        }

        // data set in class 2
        for (int i = train_N/2; i < train_N; i++) {
            train_X[i][0] = g2.random();
            train_X[i][1] = g1.random();
            train_T[i] = -1;
        }
        for (int i = test_N/2; i < test_N; i++) {
            test_X[i][0] = g2.random();
            test_X[i][1] = g1.random();
            test_T[i] = -1;
        }


		int epoch = 0;

		Perceptrons classifier = new Perceptrons(nIn);

		while (true) {
			int classified_ = 0;

			for (int i = 0; i < train_N; i++) {
				classified_ += classifier.train(train_X[i], train_T[i], learningRate);
			}

			if (classified_ == train_N) break;

			epoch++;								// 全データが正しく分類されていたら
			if (epoch > epochs) break;			// 学習をストップ
		}

		for (int i= 0; i < test_N; i++) {
			// テストデータの分類結果を配列に確認
			predicted_T[i] = classifier.predict(test_X[i]);
		}

		int[][] confusionMatrix = new int[2][2];
		double accuracy = 0.;
		double precision = 0.;
		double recall = 0.;

        for (int i = 0; i < test_N; i++) {

            if (predicted_T[i] > 0) {
                if (test_T[i] > 0) {
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusionMatrix[0][0] += 1;
                } else {
                    confusionMatrix[1][0] += 1;
                }
            } else {
                if (test_T[i] > 0) {
                    confusionMatrix[0][1] += 1;
                } else {
                    accuracy += 1;
                    confusionMatrix[1][1] += 1;
                }
            }

        }

        accuracy /= test_N;
        precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
        recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

		System.out.println("----------------------------");
		System.out.println("Perceptrons model evaluation");
		System.out.println("----------------------------");
		System.out.printf("Accurancy:  %.1f %%\n", accuracy * 100);
		System.out.printf("Precision:  %.1f %%\n", precision * 100);
		System.out.printf("Recall:     %.1f %%\n", recall * 100);

	}

}
