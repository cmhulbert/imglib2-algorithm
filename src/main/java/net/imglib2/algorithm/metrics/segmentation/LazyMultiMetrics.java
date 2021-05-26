package net.imglib2.algorithm.metrics.segmentation;

import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.type.numeric.IntegerType;

import static net.imglib2.algorithm.metrics.segmentation.SegmentationHelper.hasIntersectingLabels;

/**
 * The LazyMultiMetrics computes a running {@link MultiMetrics} over all images added by calling
 * {@link #addTimePoint(ImgLabeling, ImgLabeling) addTimePoint}. The images are expected to be
 * of dimension XYZ, where Z can be of depth 1.
 * <p>
 * The scores are calculated by accumulating the number of TP, FP, FN and the sum of the IoU for each image.
 * Each contribution is computed by adding images using {@link #addTimePoint(ImgLabeling, ImgLabeling) addTimePoint}. The final score can be queried by
 * {@link #computeScore()}.
 * <p>
 * Each image's contributions are calculated independently. Therefore, the same LazyMultiMetrics object
 * can be called from multiple threads in order to speed up computation. For instance, if the
 * total stack does not fit in memory, lazy loading and multithreading can be used to compute the
 * metrics scores by splitting the XYZ images between threads and adding them one by one. The final scores
 * can be calculated once all threads have finished.
 * <p>
 * The {@link MultiMetrics} scores are calculated at a certain {@code threshold}. This threshold is
 * the minimum IoU between a ground-truth and a prediction label at which two labels are considered
 * a potential match. The {@code threshold} can only be set at instantiation.
 *
 * @author Joran Deschamps
 * @see MultiMetrics
 */
public class LazyMultiMetrics
{
	private AtomicInteger aTP = new AtomicInteger( 0 );

	private AtomicInteger aFP = new AtomicInteger( 0 );

	private AtomicInteger aFN = new AtomicInteger( 0 );

	private AtomicLong aIoU = new AtomicLong( 0 );

	private final double threshold;

	/**
	 * Constructor with a default threshold of 0.5.
	 */
	public LazyMultiMetrics()
	{
		this.threshold = 0.5;
	}

	/**
	 * Constructor that sets the threshold value.
	 *
	 * @param threshold
	 * 		Threshold
	 */
	public LazyMultiMetrics( double threshold )
	{
		this.threshold = threshold;
	}

	/**
	 * Add a new image pair and compute its contribution to the metrics scores. The current
	 * scores can be computed by calling {@link #computeScore()}. This method is not compatible with
	 * {@link ImgLabeling} with intersecting labels.
	 *
	 * @param groundTruth
	 * 		Ground-truth image
	 * @param prediction
	 * 		Predicted image
	 * @param <T>
	 * 		Label type associated to the ground-truth
	 * @param <I>
	 * 		Ground-truth pixel type
	 * @param <U>
	 * 		Label type associated to the prediction
	 * @param <J>
	 * 		Prediction pixel type
	 */
	public < T, I extends IntegerType< I >, U, J extends IntegerType< J > > void addTimePoint(
			final ImgLabeling< T, I > groundTruth,
			final ImgLabeling< U, J > prediction
	)
	{
		if ( hasIntersectingLabels( groundTruth ) || hasIntersectingLabels( prediction ) )
			throw new UnsupportedOperationException( "ImgLabeling with intersecting labels are not supported." );

		addTimePoint( groundTruth.getIndexImg(), prediction.getIndexImg() );
	}

	/**
	 * Add a new image pair and compute its contribution to the metrics scores. The current
	 * scores can be computed by calling {@link #computeScore()}.
	 *
	 * @param groundTruth
	 * 		Ground-truth image
	 * @param prediction
	 * 		Predicted image
	 * @param <I>
	 * 		Ground-truth pixel type
	 * @param <J>
	 * 		Prediction pixel type
	 */
	public < I extends IntegerType< I >, J extends IntegerType< J > > void addTimePoint(
			RandomAccessibleInterval< I > groundTruth,
			RandomAccessibleInterval< J > prediction
	)
	{

		if ( !Arrays.equals( groundTruth.dimensionsAsLongArray(), prediction.dimensionsAsLongArray() ) )
			throw new IllegalArgumentException( "Image dimensions must match." );

		// compute multi metrics between the two images
		final MultiMetrics.MetricsSummary result = MultiMetrics.runSingle( groundTruth, prediction, threshold );

		// add results to the aggregates
		addPoint( result );
	}

	/**
	 * Compute the total {@link MultiMetrics} scores. If no image was added, or all images were empty, then the metrics
	 * scores are TP=FP=FN=0 and NaN for the others.
	 *
	 * @return Metrics scores
	 */
	public HashMap< MultiMetrics.Metrics, Double > computeScore()
	{
		MultiMetrics.MetricsSummary summary = new MultiMetrics.MetricsSummary();

		int tp = aTP.get();
		int fp = aFP.get();
		int fn = aFN.get();
		double sumIoU = atomicLongToDouble( aIoU );

		summary.addPoint( tp, fp, fn, sumIoU );

		return summary.getScores();
	}

	/**
	 * Update the atomic aggregates with the values held by the {@link net.imglib2.algorithm.metrics.segmentation.MultiMetrics.MetricsSummary}.
	 *
	 * @param newResult Result to add to the aggregates
	 */
	protected void addPoint( MultiMetrics.MetricsSummary newResult )
	{
		this.aTP.addAndGet( newResult.getTP() );
		this.aFP.addAndGet( newResult.getFP() );
		this.aFN.addAndGet( newResult.getFN() );

		addToAtomicLong( aIoU, newResult.getIoU() );
	}

	/**
	 * Add the value of {@code b} to an atomic long {@code a} representing
	 * a double value.
	 *
	 * @param a
	 * 		Atomic long to update
	 * @param b
	 * 		Value to add to the atomic long
	 */
	private void addToAtomicLong( AtomicLong a, double b )
	{
		a.set( Double.doubleToRawLongBits( Double.longBitsToDouble( a.get() ) + b ) );
	}

	/**
	 * Return the double value represented by the atomic long {@code a}.
	 *
	 * @param a
	 * 		Atomic long representing a double value
	 *
	 * @return Double value represented by {@code a}
	 */
	private double atomicLongToDouble( AtomicLong a )
	{
		return Double.longBitsToDouble( a.get() );
	}

}
