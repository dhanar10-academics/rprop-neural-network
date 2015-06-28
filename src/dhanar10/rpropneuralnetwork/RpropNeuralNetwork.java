package dhanar10.rpropneuralnetwork;

public class RpropNeuralNetwork {
	public static final double DELTA_ZERO = 0.1;
	public static final double ETA_PLUS = 1.2;
	public static final double ETA_MINUS = 0.5;
	public static final double DELTA_MAX = 50;
	public static final double DELTA_MIN = 0.000001;
	
	private double yInput[];
	private double yHidden[];
	private double yOutput[];
	
	private double wInputHidden[][];
	private double wHiddenOutput[][];
	
	private double dInputHidden[][];
	private double dHiddenOutput[][];
	
	private double dwInputHidden[][];
	private double dwHiddenOutput[][];
	
	private double gpInputHidden[][];
	private double gpHiddenOutput[][];
	
	private double targetMse = 0.0001;
	private int maxEpoch = 1000;
	
	private double mse = Double.MAX_VALUE;
	private int epoch = 0;

	public static void main(String[] args) {
		double data[][] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}; // XOR
		
		RpropNeuralNetwork rprop = new RpropNeuralNetwork(2, 2, 1);
		rprop.setTargetMse(0.0001);
		rprop.setMaxEpoch(1000);
		
		while (rprop.canTrain()) {
			rprop.train(data);
			
			System.out.println(rprop.getEpoch() + "\t" + rprop.getMse());
		}
		
		System.out.println();
		
		for (int i = 0; i < data.length; i++) {
			double output[] = rprop.calculate(data[i]);
			
			for (int j = 0; j < data[i].length - output.length; j++) {
				System.out.printf("%.2f%s", data[i][j], "\t");
			}
			
			for (int j = 0; j < output.length; j++) {
				System.out.printf("%.2f%s", output[j], j + 1 != output.length ? "\t" : "");
			}
			
			System.out.println();
		}
		
		System.exit(rprop.getMse() < rprop.getTargetMse() ? 0 : 1);
	}

	public RpropNeuralNetwork(int input, int hidden, int output) {
		yInput = new double[input + 1];
		yHidden = new double[hidden + 1];
		yOutput = new double[output];
		
		wInputHidden = new double[yInput.length][yHidden.length];
		wHiddenOutput = new double[yHidden.length][yOutput.length];
		
		dInputHidden = new double[yInput.length][yHidden.length];
		dHiddenOutput = new double[yHidden.length][yOutput.length];
		
		dwInputHidden = new double[yInput.length][yHidden.length];
		dwHiddenOutput = new double[yHidden.length][yOutput.length];
		
		gpInputHidden = new double[yInput.length][yHidden.length];
		gpHiddenOutput = new double[yHidden.length][yOutput.length];
		
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
	}
	
	public boolean train(double data[][]) {
		double gInputHidden[][] = new double[yInput.length][yHidden.length];
		double gHiddenOutput[][] = new double[yHidden.length][yOutput.length];
		
		if (!canTrain()) {
			return false;
		}
		
		mse = 0;
		
		epoch++;
		
		for (double[] d : data) {
			double yTarget[] = new double[yOutput.length];
			
			double eHidden[] = new double[yHidden.length];
			double eOutput[] = new double[yOutput.length];
			
			for (int i = 0; i < d.length; i++) {
				if (i < yInput.length - 1) {
					yInput[i] = d[i];
				}
				else {
					yTarget[i - (yInput.length - 1)] = d[i];
				}
			}
			
			yInput[yInput.length - 1] = 1;
			
			for (int i = 0; i < yHidden.length - 1; i++) {
				yHidden[i] = 0;
				
				for (int j = 0; j < yInput.length; j++) {
					yHidden[i] += yInput[j] * wInputHidden[j][i];
				}
				
				yHidden[i] = sigmoid(yHidden[i]);
			}
			
			yHidden[yHidden.length - 1] = 1;
			
			for (int i = 0; i < yOutput.length; i++) {
				yOutput[i] = 0;
				
				for (int j = 0; j < yHidden.length; j++) {
					yOutput[i] += yHidden[j] * wHiddenOutput[j][i];
				}
				
				yOutput[i] = sigmoid(yOutput[i]);
			}
			
			for (int i = 0; i < yOutput.length; i++) {
				eOutput[i] = -1 * (yTarget[i] - yOutput[i]) * dsigmoid(yOutput[i]);
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
				mse += (yTarget[j] - yOutput[j]) * (yTarget[j] - yOutput[j]);
			}
		}
		
		mse /= data.length * data[0].length;
		
		for (int i = 0; i < wInputHidden[0].length; i++) {
			for (int j = 0; j < wInputHidden.length; j++) {
				double change = gpInputHidden[j][i] * gInputHidden[j][i];
				
				if (change > 0) {
					dInputHidden[j][i] = Math.min(dInputHidden[j][i] * ETA_PLUS, DELTA_MAX);
					dwInputHidden[j][i] = -Math.signum(gInputHidden[j][i])  * dInputHidden[j][i];
					wInputHidden[j][i] += dwInputHidden[j][i];
					gpInputHidden[j][i] = gInputHidden[j][i];
				}
				else if (change < 0) {
					dInputHidden[j][i] = Math.max(dInputHidden[j][i] * ETA_MINUS, DELTA_MIN);
					gpInputHidden[j][i] = 0;
				}
				else if (change == 0) {
					dwInputHidden[j][i] = -Math.signum(gInputHidden[j][i])  * dInputHidden[j][i];
					wInputHidden[j][i] += dwInputHidden[j][i];
					gpInputHidden[j][i] = gInputHidden[j][i];
				}
			}
		}
		
		for (int i = 0; i < wHiddenOutput[0].length; i++) {
			for (int j = 0; j < wHiddenOutput.length; j++) {
				double change = gpHiddenOutput[j][i] * gHiddenOutput[j][i];
				
				if (change > 0) {
					dHiddenOutput[j][i] = Math.min(dHiddenOutput[j][i] * ETA_PLUS, DELTA_MAX);
					dwHiddenOutput[j][i] = -Math.signum(gHiddenOutput[j][i])  * dHiddenOutput[j][i];
					wHiddenOutput[j][i] += dwHiddenOutput[j][i];
					gpHiddenOutput[j][i] = gHiddenOutput[j][i];
				}
				else if (change < 0) {
					dHiddenOutput[j][i] = Math.max(dHiddenOutput[j][i] * ETA_MINUS, DELTA_MIN);
					gpHiddenOutput[j][i] = 0;
				}
				else if (change == 0) {
					dwHiddenOutput[j][i] = -Math.signum(gHiddenOutput[j][i])  * dHiddenOutput[j][i];
					wHiddenOutput[j][i] += dwHiddenOutput[j][i];
					gpHiddenOutput[j][i] = gHiddenOutput[j][i];
				}
			}
		}
		
		return true;
	}
	
	public double[] calculate(double input[]) {
		for (int i = 0; i < yInput.length - 1; i++) {
			yInput[i] = input[i];
		}
		
		yInput[yInput.length - 1] = 1;
		
		for (int i = 0; i < yHidden.length - 1; i++) {
			yHidden[i] = 0;
			
			for (int j = 0; j < yInput.length; j++) {
				yHidden[i] += yInput[j] * wInputHidden[j][i];
			}
			
			yHidden[i] = sigmoid(yHidden[i]);
		}
		
		yHidden[yHidden.length - 1] = 1;
		
		for (int i = 0; i < yOutput.length; i++) {
			yOutput[i] = 0;
			
			for (int j = 0; j < yHidden.length; j++) {
				yOutput[i] += yHidden[j] * wHiddenOutput[j][i];
			}
			
			yOutput[i] = sigmoid(yOutput[i]);
		}
		
		return yOutput;
	}
	
	public boolean canTrain() {
		return !(mse < targetMse || epoch == maxEpoch);
	}
	
	public double getTargetMse() {
		return targetMse;
	}
	
	public void setTargetMse(double targetMse) {
		this.targetMse = targetMse;
	}
	
	public double getMaxEpoch() {
		return maxEpoch;
	}
	
	public void setMaxEpoch(int maxEpoch) {
		this.maxEpoch = maxEpoch;
	}
	
	public double getMse() {
		return mse;
	}
	
	public int getEpoch() {
		return epoch;
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private double dsigmoid(double x) {
		return x * (1 - x);
	}
}
