package dhanar10.rpropneuralnetwork;

public class RpropNeuralNetwork {
	public static final double DELTA_ZERO = 0.1;
	public static final double ETA_PLUS = 1.2;
	public static final double ETA_MINUS = 0.5;
	public static final double DELTA_MAX = 50;
	public static final double DELTA_MIN = 0.000001;
	
	private double mse = 0;
	
	private double yInput[];
	private double yHidden[];
	private double yOutput[];
	
	private double wInputHidden[][];
	private double wHiddenOutput[][];

	public static void main(String[] args) {
		double data[][] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}; // XOR
		
		RpropNeuralNetwork rprop = new RpropNeuralNetwork(2, 4, 1);
		boolean success = rprop.train(data, 0.001, 10000);
		
		System.out.println();
		
		for (int i = 0; i < data.length; i++) {
			double output[] = rprop.compute(data[i]);
			
			for (int j = 0; j < data[i].length - output.length; j++) {
				System.out.printf("%.2f%s", data[i][j], "\t");
			}
			
			for (int j = 0; j < output.length; j++) {
				System.out.printf("%.2f%s", output[j], j + 1 != output.length ? "\t" : "");
			}
			
			System.out.println();
		}
		
		System.exit(success ? 0 : 1);
	}
	
	public RpropNeuralNetwork(int input, int hidden, int output) {
		yInput = new double[input];
		yHidden = new double[hidden];
		yOutput = new double[output];
		
		wInputHidden = new double[input][hidden];
		wHiddenOutput = new double[hidden][output];
	}
	
	public boolean train(double data[][], double targetMse, double maxEpoch) {
		boolean success = true;
		
		int epoch = 0;
		
		double dInputHidden[][] = new double[yInput.length][yHidden.length];
		double dHiddenOutput[][] = new double[yHidden.length][yOutput.length];
		
		double gpInputHidden[][] = new double[yInput.length][yHidden.length];
		double gpHiddenOutput[][] = new double[yHidden.length][yOutput.length];
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = Math.random() * 2 - 1;
			}
		}
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				wHiddenOutput[i][j] = Math.random() * 2 - 1;
			}
		}
		
		// BEGIN Nguyen-Widrow
		
		double bInputHidden = 0.7 * Math.pow(yHidden.length, 1.0 / yInput.length);
		double nInputHidden = 0;
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				nInputHidden += Math.pow(wInputHidden[i][j], 2);
			}
		}
		
		nInputHidden = Math.sqrt(nInputHidden);
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = bInputHidden * wInputHidden[i][j] / nInputHidden;
			}
		}
		
		double bHiddenOutput = 0.7 * Math.pow(yOutput.length, 1.0 / yHidden.length);
		double nHiddenOutput = 0;
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				nHiddenOutput += Math.pow(wHiddenOutput[i][j], 2);
			}
		}
		
		nHiddenOutput = Math.sqrt(nHiddenOutput);
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				wHiddenOutput[i][j] = bHiddenOutput * wHiddenOutput[i][j] / nHiddenOutput;
			}
		}
		
		// END Nguyen-Widrow
		
		for (int i = 0; i < dInputHidden.length; i++) {
			for (int j = 0; j < dInputHidden[0].length; j++) {
				dInputHidden[i][j] = DELTA_ZERO;
			}
		}
		
		for (int i = 0; i < dHiddenOutput.length; i++) {
			for (int j = 0; j < dHiddenOutput[0].length; j++) {
				dHiddenOutput[i][j] = DELTA_ZERO;
			}
		}
		
		while (true) {
			double gInputHidden[][] = new double[yInput.length][yHidden.length];
			double gHiddenOutput[][] = new double[yHidden.length][yOutput.length];

			epoch++;
			
			mse = 0;
			
			for (double[] d : data) {
				double yTarget[] = new double[yOutput.length];
				
				double eHidden[] = new double[yHidden.length];
				double eOutput[] = new double[yOutput.length];
				
				for (int i = 0; i < d.length; i++) {
					if (i < yInput.length) {
						yInput[i] = d[i];
					}
					else {
						yTarget[i - yInput.length] = d[i];
					}
				}
				
				this.compute(yInput);
				
				for (int i = 0; i < yOutput.length; i++) {
					eOutput[i] = (yTarget[i] - yOutput[i]) * dsigmoid(yOutput[i]);
				}
				
				for (int i = 0; i < yHidden.length; i++) {
					for (int j = 0; j < yOutput.length; j++) {
						eHidden[i] += eOutput[j] * wHiddenOutput[i][j];
					}
					
					eHidden[i] *= dsigmoid(yHidden[i]);
				}
				
				for (int i = 0; i < yHidden.length; i++) {
					for (int j = 0; j < yInput.length; j++) {
						gInputHidden[j][i] += eHidden[i] * yInput[j];
					}
				}
				
				for (int i = 0; i < yOutput.length; i++) {
					for (int j = 0; j < yHidden.length; j++) {
						gHiddenOutput[j][i] += eOutput[i] * yHidden[j];
					}
				}
				
				for (int j = 0; j < yOutput.length; j++) {
					mse += Math.pow(yTarget[j] - yOutput[j], 2);
				}
			}
			
			for (int i = 0; i < yHidden.length; i++) {
				for (int j = 0; j < yInput.length; j++) {
					double change = gpInputHidden[j][i] * gInputHidden[j][i];
					
					if (change > 0) {
						dInputHidden[j][i] = Math.min(dInputHidden[j][i] * ETA_PLUS, DELTA_MAX);
						wInputHidden[j][i] += Math.signum(gInputHidden[j][i])  * dInputHidden[j][i];
						gpInputHidden[j][i] = gInputHidden[j][i];
					}
					else if (change < 0) {
						dInputHidden[j][i] = Math.max(dInputHidden[j][i] * ETA_MINUS, DELTA_MIN);
						gpInputHidden[j][i] = 0;
					}
					else if (change == 0) {
						wInputHidden[j][i] += Math.signum(gInputHidden[j][i]) * dInputHidden[j][i];
						gpInputHidden[j][i] = gInputHidden[j][i];
					}
				}
			}
			
			for (int i = 0; i < yOutput.length; i++) {
				for (int j = 0; j < yHidden.length; j++) {
					double change = gpHiddenOutput[j][i] * gHiddenOutput[j][i];
					
					if (change > 0) {
						dHiddenOutput[j][i] = Math.min(dHiddenOutput[j][i] * ETA_PLUS, DELTA_MAX);
						wHiddenOutput[j][i] += Math.signum(gHiddenOutput[j][i])  * dHiddenOutput[j][i];
						gpHiddenOutput[j][i] = gHiddenOutput[j][i];
					}
					else if (change < 0) {
						dHiddenOutput[j][i] = Math.max(dHiddenOutput[j][i] * ETA_MINUS, DELTA_MIN);
						gpHiddenOutput[j][i] = 0;
					}
					else if (change == 0) {
						wHiddenOutput[j][i] += Math.signum(gHiddenOutput[j][i]) * dHiddenOutput[j][i];
						gpHiddenOutput[j][i] = gHiddenOutput[j][i];
					}
				}
			}
			
			mse /= data.length * data[0].length;
			
			System.out.println(epoch + "\t" + mse);
			
			if (mse < targetMse) {
				break;
			}
			
			if (epoch == maxEpoch) {
				success = false;;
				break;
			}
		}
		
		return success;
	}
	
	public double[] compute(double input[]) {
		for (int i = 0; i < yInput.length; i++) {
			yInput[i] = input[i];
		}
		
		for (int i = 0; i < yHidden.length; i++) {
			yHidden[i] = 0;
			
			for (int j = 0; j < yInput.length; j++) {
				yHidden[i] += yInput[j] * wInputHidden[j][i];
			}
			
			yHidden[i] = sigmoid(yHidden[i]);
		}
		
		for (int i = 0; i < yOutput.length; i++) {
			yOutput[i] = 0;
			
			for (int j = 0; j < yHidden.length; j++) {
				yOutput[i] += yHidden[j] * wHiddenOutput[j][i];
			}
			
			yOutput[i] = sigmoid(yOutput[i]);
		}
		
		return yOutput;
	}
	
	public double getMse() {
		return mse;
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private double dsigmoid(double x) {
		return x * (1 - x);
	}
}
