package net.imglib2.algorithm.mgdm;

import de.mpg.cbs.structures.BinaryHeap2D;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.type.logic.NativeBoolType;
import net.imglib2.type.numeric.integer.GenericIntType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.view.Views;

import java.util.Arrays;
import java.util.function.Predicate;

public class ReferenceMGDM<B extends GenericIntType<B>> {

	// data and membership buffers
	private RandomAccessible<FloatType>[] mgdmFunctions;    // MGDM's pseudo level set mgdmfunctions
	private RandomAccessible<B>[] mgdmLabels;        // MGDM's label maps
	private byte nmgdm;                    // total number of MGDM mgdmlabels and mgdmfunctions
	private final int[] dimensions;
	private final double[] resolution;

	// atlas parameters
	private int nobj;        // number of shapes
	private byte[] objLabel;            // label values in the original image

	// data and membership buffers
	private RandomAccessible<IntType> segmentation;    // MGDM's segmentation
	private RandomAccessible<UnsignedIntType> counter;
	private RandomAccessible<NativeBoolType> mask;                // masking regions not used in computations
	private BinaryHeap2D heap;                // the heap used in fast marching

	// parameters
	private float maxDist = 1e9f;
	private short maxcount = 5;
	private boolean stopDist = false;
	private boolean skipZero = false;

	// computation variables to avoid re-allocating

	// for minimumMarchingDistance
	double s, s2, tmp;
	int count;
	double dist;

	// useful constants & flags
	private static final byte EMPTY = -1;

	// neighborhood flags (ordering is important!!!)
	public static final byte pX = 0;
	public static final byte pY = 1;
	public static final byte pZ = 2;
	public static final byte mX = 3;
	public static final byte mY = 4;
	public static final byte mZ = 5;
	public static final byte pXpY = 6;
	public static final byte pYpZ = 7;
	public static final byte pZpX = 8;
	public static final byte pXmY = 9;
	public static final byte pYmZ = 10;
	public static final byte pZmX = 11;
	public static final byte mXpY = 12;
	public static final byte mYpZ = 13;
	public static final byte mZpX = 14;
	public static final byte mXmY = 15;
	public static final byte mYmZ = 16;
	public static final byte mZmX = 17;
	public static final byte pXpYpZ = 18;
	public static final byte pXmYmZ = 19;
	public static final byte pXmYpZ = 20;
	public static final byte pXpYmZ = 21;
	public static final byte mXpYpZ = 22;
	public static final byte mXmYmZ = 23;
	public static final byte mXmYpZ = 24;
	public static final byte mXpYmZ = 25;
	public static final byte NGB = 26;
	public static final byte CTR = 26;

	private static final byte[] ngbx = {+1, 0, 0, -1, 0, 0, +1, 0, +1, +1, 0, -1, -1, 0, +1, -1, 0, -1, +1, +1, +1, +1, -1, -1, -1, -1};
	private static final byte[] ngby = {0, +1, 0, 0, -1, 0, +1, +1, 0, -1, +1, 0, +1, -1, 0, -1, -1, 0, +1, -1, -1, +1, +1, -1, -1, +1};
	private static final byte[] ngbz = {0, 0, +1, 0, 0, -1, 0, +1, +1, 0, -1, +1, 0, +1, -1, 0, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1};

	private static int[] xoff = new int[]{ 1, -1, 0, 0, 0, 0 };
	private static int[] yoff = new int[]{ 0, 0, 1, -1, 0, 0 };
	private static int[] zoff = new int[]{ 0, 0, 0, 0, 1, -1 };

	// numerical quantities
	private static final float INF = 1e15f;
	private static final float ZERO = 1e-15f;
	private static final float PI2 = (float)(Math.PI / 2.0);
	private final static float SQR2 = (float)Math.sqrt(2.0f);
	private final static float SQR3 = (float)Math.sqrt(3.0f);
	private final static float diagdist = 1 / (2 * SQR2);
	private final static float cubedist = 1 / (2 * SQR3);
	private static final float UNKNOWN = -1.0f;

	// for debug and display
	private static final boolean debug = false;
	private static final boolean verbose = true;

	public static void main(String[] args) {

		double[] res = new double[]{1, 1, 1};

		final int[] labels = new int[]{
				2, 2, 1, 		 2  , -1 , 2  ,		 3  , -1 , 3  ,
				2, 1, 2, 		 -1 , -1 , -1 ,		 -1 , -1 , -1 ,
				1, 2, 1, 		 2  , -1 , 2  ,		 3  , -1 , 3  ,
		};

		final ArrayImg<IntType, IntArray> labeledImg = ArrayImgs.ints(labels, 3, 3);

		final int nmgdm = 2;

		final ReferenceMGDM<IntType> test = new ReferenceMGDM<>(
				labeledImg,
				null,
				res,
				new byte[]{1, 2, 3},
				3,
				nmgdm,
				false,
				-1,
				it -> it.getInteger() != -1,
				new IntType(-1)
		);
		for (long x = 0; x < labeledImg.dimension(0); x++) {
			for (long y = 0; y < labeledImg.dimension(1); y++) {
				for (long z = 0; z < labeledImg.dimension(2); z++) {
					final StringBuilder distLabels = new StringBuilder();
					distLabels.append("(");
					for (int n = 0; n < nmgdm; n++) {
						distLabels.append(test.mgdmFunctions[n].getAt(x, y, z));
						distLabels.append(":");
						distLabels.append(test.mgdmLabels[n].getAt(x, y, z));
						if (n < nmgdm - 1)
							distLabels.append(", ");
					}
					distLabels.append(")");
					System.out.print(distLabels);
				}
				System.out.print(",\t");
			}
			System.out.println("");
		}

	}

	/**
	 * constructors
	 */
	public ReferenceMGDM(
			RandomAccessibleInterval<B> segmentation_,
			float[] boundary,
			double[] resolution,
			byte[] labels,
			int numLabels,
			int nmgdm,
			boolean skip,
			float dist,
			Predicate<B> acceptLabel,
			B invalidLabel) {

		final RandomAccessibleInterval<B> segmentation3D;
		if (segmentation_.numDimensions() == 3) {
			segmentation3D = segmentation_;
		} else if (segmentation_.numDimensions() == 2) {
			segmentation3D = Views.addDimension(segmentation_, 0, 0);
		} else {
			throw new IllegalArgumentException("Segmentation must be 2D or 3D");
		}

		this.dimensions = Arrays.stream(segmentation3D.dimensionsAsLongArray()).mapToInt(it -> (int)it).toArray();
		this.resolution = Arrays.copyOf(resolution, resolution.length);

		nobj = numLabels;
		objLabel = labels;

		this.nmgdm = (byte)nmgdm;
		skipZero = skip;
		stopDist = (dist == -1);
		if (!stopDist)
			maxDist = dist;

		maxcount = (short)Math.max(maxcount, nmgdm);

		// init all the arrays
		try {
			segmentation = Views.extendValue(ArrayImgs.ints(segmentation3D.dimensionsAsLongArray()), invalidLabel.getIntegerLong());
			counter = Views.extendValue(ArrayImgs.unsignedInts(segmentation3D.dimensionsAsLongArray()), -1);
			mask = Converters.convert(
					Views.extendValue(segmentation3D, invalidLabel),
					counter,
					(label, count, maskVal) -> maskVal.set(count.get() > -1 && count.get() < this.nmgdm && acceptLabel.test(label)),
					new NativeBoolType(true)
			);

			mgdmFunctions = new RandomAccessible[nmgdm];
			for (int i = 0; i < mgdmFunctions.length; i++) {
				mgdmFunctions[i] = Views.extendValue(ArrayImgs.floats(segmentation3D.dimensionsAsLongArray()), maxDist);
			}

			mgdmLabels = new RandomAccessible[nmgdm + 1];
			for (int i = 0; i < mgdmLabels.length; i++) {
				final ArrayImg<B, ?> labelImg = new ArrayImgFactory<>(invalidLabel).create(dimensions);
				mgdmLabels[i] = Views.extendValue(labelImg, invalidLabel.getInteger());
			}
			// initalize the heap too so we don't have to do it multiple times
			heap = new BinaryHeap2D(dimensions[0] * dimensions[1] + dimensions[1] * dimensions[2] + dimensions[2] * dimensions[0],
					BinaryHeap2D.MINTREE);
		} catch (OutOfMemoryError e) {
			finalize();
			System.out.println(e.getMessage());
			return;
		}

		// init segmentation (from atlas if null)
		for (int x = 0; x < dimensions[0]; x++)
			for (int y = 0; y < dimensions[1]; y++)
				for (int z = 0; z < dimensions[2]; z++) {

					int lb = invalidLabel.getInteger();
					for (byte n = 0; n < nobj; n++) {
						if (objLabel[n] == segmentation3D.getAt(x, y, z).getInteger()) {
							lb = objLabel[n];
							continue;
						}
					}
					segmentation.getAt(x, y, z).set(lb);

					// init the MGDM model
					mgdmLabels[0].getAt(x, y, z).setInteger(lb);
					if (boundary != null)
						mgdmFunctions[0].getAt(x, y, z).set(boundary[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z]);
					else
						mgdmFunctions[0].getAt(x, y, z).set(0.5f);
				}
		// build the full model
		fastMarchingReinitialization(stopDist, false, true);

	}

//	/**
//	 * constructors
//	 */
//	public ReferenceMGDM(int[] segmentation_, float[] bound,
//			int[] dimensions, double[] resolution,
//			byte[] labels, int numLabels,
//			int nmgdm, boolean skip, float dist) {
//
//		this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
//		this.resolution = Arrays.copyOf(resolution, resolution.length);
//
//		nobj = numLabels;
//		objLabel = labels;
//
//		this.nmgdm = (byte)nmgdm;
//		skipZero = skip;
//		stopDist = (dist == -1);
//		if (!stopDist)
//			maxDist = dist;
//
//		maxcount = (short)Math.max(maxcount, this.nmgdm);
//
//		// 6-neighborhood: pre-compute the index offsets
//		xoff = new long[]{1, -1, 0, 0, 0, 0};
//		yoff = new long[]{0, 0, dimensions[0], -dimensions[0], 0, 0};
//		zoff = new long[]{0, 0, 0, 0, dimensions[0] * dimensions[1], -dimensions[0] * dimensions[1]};
//
//		// init all the arrays
//		try {
//			segmentation = new byte[dimensions[0] * dimensions[1] * dimensions[2]];
//			mask = new boolean[dimensions[0] * dimensions[1] * dimensions[2]];
//			counter = new short[dimensions[0] * dimensions[1] * dimensions[2]];
//			mgdmfunctions = new float[this.nmgdm][dimensions[0] * dimensions[1] * dimensions[2]];
//			mgdmlabels = new byte[this.nmgdm + 1][dimensions[0] * dimensions[1] * dimensions[2]];
//			// initalize the heap too so we don't have to do it multiple times
//			heap = new BinaryHeap2D(dimensions[0] * dimensions[1] + dimensions[1] * dimensions[2] + dimensions[2] * dimensions[0], BinaryHeap2D.MINTREE);
//		} catch (OutOfMemoryError e) {
//			finalize();
//			System.out.println(e.getMessage());
//			return;
//		}
//		// basic mask: remove two layers off the images (for avoiding limits)
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++) {
//					if (x > 1 && x < dimensions[0] - 2 && y > 1 && y < dimensions[1] - 2 && z > 1 && z < dimensions[2] - 2)
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = true;
//					else
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = false;
//					if (skipZero && segmentation_[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] == 0)
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = false;
//				}
//		// init segmentation (from atlas if null)
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++) {
//					int xyz = x + dimensions[0] * y + dimensions[0] * dimensions[1] * z;
//
//					byte nlb = EMPTY;
//					for (byte n = 0; n < nobj; n++) {
//						if (objLabel[n] == segmentation_[xyz]) {
//							nlb = n;
//							continue;
//						}
//					}
//					segmentation[xyz] = nlb;
//					counter[xyz] = 0;
//
//					// init the MGDM model
//					mgdmlabels[0][xyz] = nlb;
//					if (bound != null)
//						mgdmfunctions[0][xyz] = bound[xyz];
//					else
//						mgdmfunctions[0][xyz] = 0.5f;
//				}
//		// build the full model
//		fastMarchingReinitialization(stopDist, false, true);
//
//	}
//
//	/**
//	 * constructors
//	 */
//	public ReferenceMGDM(byte[][][] segmentation_, float[][][] bound,
//			int[] dimensions, double[] resolution,
//			byte[] labels, int numLabels,
//			int nmgdm, boolean skip, float dist) {
//
//		this.dimensions = Arrays.copyOf(dimensions, dimensions.length);
//		this.resolution = Arrays.copyOf(resolution, resolution.length);
//
//		nobj = numLabels;
//		objLabel = labels;
//
//		this.nmgdm = (byte)nmgdm;
//		skipZero = skip;
//		stopDist = (dist == -1);
//		if (!stopDist)
//			maxDist = dist;
//
//		maxcount = (short)Math.max(maxcount, this.nmgdm);
//
//		// 6-neighborhood: pre-compute the index offsets
//		xoff = new long[]{1, -1, 0, 0, 0, 0};
//		yoff = new long[]{0, 0, dimensions[0], -dimensions[0], 0, 0};
//		zoff = new long[]{0, 0, 0, 0, dimensions[0] * dimensions[1], -dimensions[0] * dimensions[1]};
//
//		// init all the arrays
//		try {
//			segmentation = new byte[dimensions[0] * dimensions[1] * dimensions[2]];
//			mask = new boolean[dimensions[0] * dimensions[1] * dimensions[2]];
//			counter = new short[dimensions[0] * dimensions[1] * dimensions[2]];
//			mgdmfunctions = new float[this.nmgdm][dimensions[0] * dimensions[1] * dimensions[2]];
//			mgdmlabels = new byte[this.nmgdm + 1][dimensions[0] * dimensions[1] * dimensions[2]];
//			// initalize the heap too so we don't have to do it multiple times
//			heap = new BinaryHeap2D(dimensions[0] * dimensions[1] + dimensions[1] * dimensions[2] + dimensions[2] * dimensions[0], BinaryHeap2D.MINTREE);
//		} catch (OutOfMemoryError e) {
//			finalize();
//			System.out.println(e.getMessage());
//			return;
//		}
//		// basic mask: remove two layers off the images (for avoiding limits)
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++) {
//					if (x > 1 && x < dimensions[0] - 2 && y > 1 && y < dimensions[1] - 2 && z > 1 && z < dimensions[2] - 2)
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = true;
//					else
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = false;
//					if (skipZero && segmentation_[x][y][z] == 0)
//						mask[x + dimensions[0] * y + dimensions[0] * dimensions[1] * z] = false;
//				}
//		// init segmentation (from atlas if null)
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++) {
//					int xyz = x + dimensions[0] * y + dimensions[0] * dimensions[1] * z;
//
//					byte nlb = EMPTY;
//					for (byte n = 0; n < nobj; n++) {
//						if (objLabel[n] == segmentation_[x][y][z]) {
//							nlb = n;
//							continue;
//						}
//					}
//					segmentation[xyz] = nlb;
//					counter[xyz] = 0;
//
//					// init the MGDM model
//					mgdmlabels[0][xyz] = nlb;
//					if (bound != null)
//						mgdmfunctions[0][xyz] = bound[x][y][z];
//					else
//						mgdmfunctions[0][xyz] = 0.5f;
//				}
//		// build the full model
//		fastMarchingReinitialization(stopDist, false, true);
//
//	}

	@Override public void finalize() {

		mgdmFunctions = null;
		mgdmLabels = null;
		segmentation = null;
		heap = null;
	}

	/**
	 * clean up the computation arrays
	 */
	public final void cleanUp() {

		mgdmFunctions = null;
		mgdmLabels = null;
		heap.finalize();
		heap = null;
		System.gc();
	}

	public final RandomAccessible<FloatType>[] getFunctions() {

		return mgdmFunctions;
	}

	public final RandomAccessible<B>[] getLabels() {

		return mgdmLabels;
	}

	public final RandomAccessible<IntType> getSegmentation() {

		return segmentation;
	}

	public final RandomAccessible<NativeBoolType> getMask() {

		return mask;
	}

//	public final void reduceMGDMsize(int nred) {
//
//		float[][] redfunctions = new float[nred][dimensions[0] * dimensions[1] * dimensions[2]];
//		byte[][] redlabels = new byte[nred + 1][dimensions[0] * dimensions[1] * dimensions[2]];
//
//		for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//			for (byte n = 0; n < nred; n++) {
//				redfunctions[n][xyz] = mgdmfunctions[n][xyz];
//				redlabels[n][xyz] = mgdmlabels[n][xyz];
//			}
//			redlabels[nred][xyz] = mgdmlabels[nred][xyz];
//		}
//
//		// replace the maps
//		nmgdm = (byte)nred;
//		mgdmfunctions = redfunctions;
//		mgdmlabels = redlabels;
//	}
//
//	/**
//	 * reconstruct the level set functions with possible approximation far from contour
//	 */
//	public final float[][][] exportBinaryLevelSet(boolean[] inside) {
//
//		float[][][] levelset = new float[dimensions[0]][dimensions[1]][dimensions[2]];
//
//		float maximumDist = maxDist;
//
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++) {
//					int xyz = x + dimensions[0] * y + dimensions[0] * dimensions[1] * z;
//
//					levelset[x][y][z] = 0.0f;
//					if (mgdmlabels[0][xyz] > -1) {
//						if (inside[mgdmlabels[0][xyz]]) {
//							// search for the next outside value, use constant if none
//							int nout = -1;
//							for (int n = 1; n < nmgdm && nout == -1; n++) {
//								if (mgdmlabels[n][xyz] > -1 && !inside[mgdmlabels[n][xyz]])
//									nout = n;
//							}
//							if (nout > -1) {
//								for (int n = 0; n < nout; n++) {
//									levelset[x][y][z] -= mgdmfunctions[n][xyz];
//								}
//							} else {
//								levelset[x][y][z] = -maximumDist;
//							}
//						} else {
//							// search for the next inside value, use constant if none
//							int nin = -1;
//							for (int n = 1; n < nmgdm && nin == -1; n++) {
//								if (mgdmlabels[n][xyz] > -1 && inside[mgdmlabels[n][xyz]])
//									nin = n;
//							}
//							if (nin > -1) {
//								for (int n = 0; n < nin; n++) {
//									levelset[x][y][z] += mgdmfunctions[n][xyz];
//								}
//							} else {
//								levelset[x][y][z] = maximumDist;
//							}
//						}
//					} else {
//						// outside by default
//						levelset[x][y][z] = maximumDist;
//					}
//				}
//		return levelset;
//	}
//
//	/**
//	 * reconstruct the level set functions with possible approximation far from contour
//	 */
//	public final float[][] reconstructedLevelSets() {
//
//		float[][] levelsets = new float[nobj][dimensions[0] * dimensions[1] * dimensions[2]];
//		for (int n = 0; n < nobj; n++) {
//			for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//				if (mgdmlabels[0][xyz] == n)
//					levelsets[n][xyz] = -mgdmfunctions[0][xyz];
//				else
//					levelsets[n][xyz] = 0.0f;
//
//				for (int l = 0; l < nmgdm && mgdmlabels[l][xyz] != n; l++) {
//					levelsets[n][xyz] += mgdmfunctions[l][xyz];
//				}
//			}
//		}
//		return levelsets;
//	}
//
//	/**
//	 * reconstruct the levelset only where it is guaranteed to be exact
//	 */
//	public final float[][] reconstructedExactLevelSets() {
//
//		float[][] levelsets = new float[nobj][dimensions[0] * dimensions[1] * dimensions[2]];
//		for (int n = 0; n < nobj; n++) {
//			for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//				if (mgdmlabels[0][xyz] == n)
//					levelsets[n][xyz] = -mgdmfunctions[0][xyz];
//				else
//					levelsets[n][xyz] = 0.0f;
//
//				int max = 0;
//				for (int l = 0; l < nmgdm && mgdmlabels[l][xyz] != n; l++) {
//					levelsets[n][xyz] += mgdmfunctions[l][xyz];
//					max++;
//				}
//				if (max == nmgdm)
//					levelsets[n][xyz] = UNKNOWN;
//			}
//		}
//		return levelsets;
//	}
//
//	/**
//	 * reconstruct the level set functions with possible approximation far from contour
//	 */
//	public final float reconstructedLevelSetAt(int xyz, byte rawlb) {
//
//		float levelset;
//
//		if (mgdmlabels[0][xyz] > -1 && mgdmlabels[0][xyz] == rawlb)
//			levelset = -mgdmfunctions[0][xyz];
//		else
//			levelset = 0.0f;
//
//		for (int l = 0; l < nmgdm && mgdmlabels[l][xyz] > -1 && mgdmlabels[l][xyz] != rawlb; l++) {
//			levelset += mgdmfunctions[l][xyz];
//		}
//		return levelset;
//	}
//
//	/**
//	 * reconstruct the level set functions with possible approximation far from contour
//	 */
//	public final int[][] reconstructedLabels() {
//
//		int[][] labels = new int[nmgdm][dimensions[0] * dimensions[1] * dimensions[2]];
//		for (int n = 0; n < nmgdm; n++) {
//			for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//				if (mgdmlabels[n][xyz] > -1) {
//					labels[n][xyz] = objLabel[mgdmlabels[n][xyz]];
//				}
//			}
//		}
//		return labels;
//	}
//
//	/**
//	 * get a final segmentation
//	 */
//	public final float[] exportSegmentation() {
//
//		float[] seg = new float[dimensions[0] * dimensions[1] * dimensions[2]];
//		for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//			//if (mgdmlabels[0][xyz]>-1) {
//			if (segmentation[xyz] > -1) {
//				//seg[xyz] = objLabel[mgdmlabels[0][xyz]];
//				seg[xyz] = objLabel[segmentation[xyz]];
//			}
//		}
//		return seg;
//	}
//
//	public final float[] exportCounter() {
//
//		float[] ct = new float[dimensions[0] * dimensions[1] * dimensions[2]];
//		for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//			ct[xyz] = counter[xyz];
//		}
//		return ct;
//	}
//
//	public final byte[][][][] exportLabels() {
//
//		byte[][][][] seg = new byte[dimensions[0]][dimensions[1]][dimensions[2]][nmgdm + 1];
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++)
//					for (int n = 0; n <= nmgdm; n++) {
//						int xyz = x + dimensions[0] * y + dimensions[0] * dimensions[1] * z;
//						if (mgdmlabels[n][xyz] > -1) {
//							seg[x][y][z][n] = objLabel[mgdmlabels[n][xyz]];
//						} else {
//							seg[x][y][z][n] = -1;
//						}
//					}
//		return seg;
//	}
//
//	public final float[][][][] exportFunctions() {
//
//		float[][][][] fun = new float[dimensions[0]][dimensions[1]][dimensions[2]][nmgdm];
//		for (int x = 0; x < dimensions[0]; x++)
//			for (int y = 0; y < dimensions[1]; y++)
//				for (int z = 0; z < dimensions[2]; z++)
//					for (int n = 0; n < nmgdm; n++) {
//						int xyz = x + dimensions[0] * y + dimensions[0] * dimensions[1] * z;
//						fun[x][y][z][n] = mgdmfunctions[n][xyz];
//					}
//		return fun;
//	}

	/**
	 * perform joint reinitialization for all labels
	 */
	public final void fastMarchingReinitialization(boolean narrowBandOnly, boolean almostEverywhere, boolean stopCounter) {
		// computation variables
		float[] nbdist = new float[6];
		boolean[] nbflag = new boolean[6];
		float curdist, newdist;
		boolean done, isprocessed;

		// compute the neighboring labels and corresponding distance functions (! not the MGDM functions !)

		long start_time = System.currentTimeMillis();

		heap.reset();
		// initialize the heap from boundaries
		for (int x = 0; x < dimensions[0]; x++) {
			for (int y = 0; y < dimensions[1]; y++) {
				for (int z = 0; z < dimensions[2]; z++) {
					if (mask.getAt(x, y, z).get()) {
						// mgdm functions : reinit everiywhere
						for (int n = 0; n < nmgdm; n++) {
							if (n > 0)
								mgdmFunctions[n].getAt(x, y, z).set(UNKNOWN);
							mgdmLabels[n].getAt(x, y, z).setInteger(EMPTY);
						}
						mgdmLabels[nmgdm].getAt(x, y, z).setInteger(EMPTY);
						// not needed, should be kept the same
						//segmentation.getAt(x, y, z).get()) = mgdmlabels[0].getAt(x, y, z).get());
						// search for boundaries
						for (int k = 0; k < 6; k++) {
							final int neighbor = segmentation.getAt(x + xoff[k], y + yoff[k], z + zoff[k]).get();
							if (neighbor != segmentation.getAt(x, y, z).get()) {
								final boolean neighborMask = mask.getAt(x + xoff[k], y + yoff[k], z + zoff[k]).get();
								if (neighborMask) {
									// add to the heap with previous value
									final float mgdnFunction = mgdmFunctions[0].getAt(x + xoff[k], y + yoff[k], z + zoff[k]).get();
									final int[] xyzNeighbor = new int[]{x + xoff[k], y + yoff[k], z + zoff[k]};
									final int neighborIndex = IntervalIndexer.positionToIndex(xyzNeighbor, dimensions);
									heap.addValue(mgdnFunction, neighborIndex, segmentation.getAt(xyzNeighbor).getBigInteger().byteValue());
								}
							}
						}
					}
				}
			}
		}

		// grow the labels and functions
		final int[] pos = new int[3];
		final int[] neighborPos = new int[3];
		final int[] neighborPosB = new int[3];
		while (heap.isNotEmpty()) {
			// extract point with minimum distance
			curdist = heap.getFirst();
			int xyz = heap.getFirstId();
			byte lb = heap.getFirstState();
			IntervalIndexer.indexToPosition(xyz, dimensions, pos);
			heap.removeFirst();

			final UnsignedIntType countAtPosType = counter.getAt(pos);
			final int countAtPos = countAtPosType.getInteger();

			// if more than nmgdm labels have been found already, this is done
			if (!mask.getAt(pos).get())
				continue;

			// if there is already a label for this object, this is done
			done = false;
			for (int n = 0; n < countAtPos; n++)
				if (mgdmLabels[n].getAt(pos).getInteger() == lb)
					done = true;
			if (done)
				continue;

			// update the distance functions at the current level
			mgdmFunctions[countAtPos].getAt(pos).set(curdist);
			mgdmLabels[countAtPos].getAt(pos).setInteger(lb);
			countAtPosType.inc(); // update the current level

			// find new neighbors
			for (int k = 0; k < 6; k++) {
				for (int i = 0; i < neighborPos.length; i++) {
					neighborPos[i] = pos[i] + xoff[k] + yoff[k] + zoff[k];
				}

				final UnsignedIntType neighborCountType = counter.getAt(neighborPos);
				final long neighborCount = neighborCountType.get();
				if (mask.getAt(neighborPos).get() && (!stopCounter || neighborCount < 2 * maxcount)) {
					// must be in outside the object or its processed neighborhood
					isprocessed = false;
					if (segmentation.getAt(neighborPos).get() == lb)
						isprocessed = true;
					else {
						for (int n = 0; n < neighborCount; n++)
							if (mgdmLabels[n].getAt(neighborPos).getInteger() == lb)
								isprocessed = true;
					}

					if (!isprocessed) {
						// compute new distance based on processed neighbors for the same object
						for (int l = 0; l < 6; l++) {
							nbdist[l] = UNKNOWN;
							nbflag[l] = false;

							for (int i = 0; i < neighborPosB.length; i++) {
								neighborPosB[i] = neighborPos[i] + xoff[k] + yoff[k] + zoff[k];
							}
							// note that there is at most one value used here
							final UnsignedIntType neighborBCountType = counter.getAt(neighborPosB);
							final long neighborBCount = neighborBCountType.get();
							for (int n = 0; n < neighborBCount; n++)
								if (mask.getAt(neighborPosB).get())
									if (mgdmLabels[n].getAt(neighborPosB).getInteger() == lb) {
										nbdist[l] = mgdmFunctions[n].getAt(neighborPosB).get();
										nbflag[l] = true;
									}
						}
						newdist = minimumMarchingDistance(nbdist, nbflag);

						if ((!narrowBandOnly && !almostEverywhere)
								|| (narrowBandOnly && newdist <= maxDist)
								|| (almostEverywhere && (segmentation.getAt(neighborPos).get() != 0 || newdist <= maxDist))) {
							// add to the heap
							final int xyzNeighbor = IntervalIndexer.positionToIndex(neighborPos, dimensions);
							heap.addValue(newdist, xyzNeighbor, lb);
						}
					}
				}
			}
		}

		// to create the MGDM functions, we need to copy the segmentation, forget the last labels
		// and compute differences between distance functions
		for (int x = 0; x < dimensions[0]; x++) {
			for (int y = 0; y < dimensions[1]; y++) {
				for (int z = 0; z < dimensions[2]; z++) {
					if (mask.getAt(x, y, z).get()) {
						// label permutation
						for (int n = nmgdm; n > 0; n--) {
							final int prevLabel = mgdmLabels[n - 1].getAt(x, y, z).getInteger();
							mgdmLabels[n].getAt(x, y, z).setInteger(prevLabel);
						}
						final int segmentationLabel = segmentation.getAt(x, y, z).get();
						mgdmLabels[0].getAt(x, y, z).setInteger(segmentationLabel);

						// distance function difference
						for (int n = nmgdm - 1; n > 0; n--) {
							final float max = Math.max(UNKNOWN, mgdmFunctions[n].getAt(x, y, z).get() - mgdmFunctions[n - 1].getAt(x, y, z).get());
							mgdmFunctions[n].getAt(x, y, z).set(max);
						}
					}
				}
			}
		}
	}

	/**
	 * the Fast marching distance computation
	 * (!assumes a 6D array with opposite coordinates stacked one after the other)
	 */
	public final float minimumMarchingDistance(float[] val, boolean[] flag) {

		// s = a + b +c; s2 = a*a + b*b +c*c
		s = 0;
		s2 = 0;
		count = 0;

		for (int n = 0; n < 6; n += 2) {
			if (flag[n] && flag[n + 1]) {
				tmp = Math.min(val[n], val[n + 1]); // Take the smaller one if both are processed
				s += tmp;
				s2 += tmp * tmp;
				count++;
			} else if (flag[n]) {
				s += val[n]; // Else, take the processed one
				s2 += val[n] * val[n];
				count++;
			} else if (flag[n + 1]) {
				s += val[n + 1];
				s2 += val[n + 1] * val[n + 1];
				count++;
			}
		}
		// count must be greater than zero since there must be at least one processed pt in the neighbors

		tmp = (s + Math.sqrt((s * s - count * (s2 - 1.0f)))) / count;

		// The larger root
		return (float)tmp;
	}

	/**
	 * the isosurface distance computation
	 * (!assumes a 6D array with opposite coordinates stacked one after the other)
	 * (the input values are all positive, the flags are true only if the isosurface crosses)
	 */
	public final float isoSurfaceDistance(double cur, float[] val, boolean[] flag) {

		if (cur == 0)
			return 0;

		s = 0;
		dist = 0;

		for (int n = 0; n < 6; n += 2) {
			if (flag[n] && flag[n + 1]) {
				tmp = Math.max(val[n], val[n + 1]); // Take the largest distance (aka closest to current point) if both are across the boundariy
				s = cur / (cur + tmp);
				dist += 1.0 / (s * s);
			} else if (flag[n]) {
				s = cur / (cur + val[n]); // Else, take the boundariy point
				dist += 1.0 / (s * s);
			} else if (flag[n + 1]) {
				s = cur / (cur + val[n + 1]);
				dist += 1.0 / (s * s);
			}
		}
		// triangular (tetrahedral?) relationship of height in right triangles gives correct distance
		tmp = Math.sqrt(1.0 / dist);

		// The larger root
		return (float)tmp;
	}

//	/**
//	 * isosurface distance re-initialization at the boundariy
//	 */
//	private final void resetIsosurfaceBoundary() {
//
//		if (debug)
//			System.out.print("fast marching evolution: iso-surface reinit\n");
//
//		float[] nbdist = new float[6];
//		boolean[] nbflag = new boolean[6];
//		boolean boundary;
//		float[] tmp = new float[dimensions[0] * dimensions[1] * dimensions[2]];
//		boolean[] processed = new boolean[dimensions[0] * dimensions[1] * dimensions[2]];
//
//		for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++)
//			if (mask[xyz]) {
//
//				boundary = false;
//				for (int l = 0; l < 6; l++) {
//					nbdist[l] = UNKNOWN;
//					nbflag[l] = false;
//
//					int xyznb = xyz + xoff[l] + yoff[l] + zoff[l];
//					if (segmentation[xyznb] != segmentation[xyz] && mask[xyznb]) {
//						// compute new distance based on processed neighbors for the same object
//						nbdist[l] = Math.abs(mgdmfunctions[0][xyznb]);
//						nbflag[l] = true;
//						boundary = true;
//					}
//				}
//				if (boundary) {
//					tmp[xyz] = isoSurfaceDistance(mgdmfunctions[0][xyz], nbdist, nbflag);
//					processed[xyz] = true;
//				}
//			}
//		// once all the new values are computed, copy into original GDM function (sign is not important here)
//		for (int xyz = 0; xyz < dimensions[0] * dimensions[1] * dimensions[2]; xyz++) {
//			if (processed[xyz])
//				mgdmfunctions[0][xyz] = tmp[xyz];
//			else
//				mgdmfunctions[0][xyz] = UNKNOWN;
//		}
//
//		return;
//	}

}
