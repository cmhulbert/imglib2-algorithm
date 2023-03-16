package net.imglib2.algorithm.mgdm;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import de.mpg.cbs.structures.BinaryHeap2D;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.array.ArrayRandomAccess;
import net.imglib2.img.basictypeaccess.array.BooleanArray;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.outofbounds.OutOfBounds;
import net.imglib2.type.BooleanType;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.logic.NativeBoolType;
import net.imglib2.type.numeric.NumericType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.ConstantUtils;
import net.imglib2.util.IntervalIndexer;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class DistanceInit< B extends BooleanType<B>> {
	
	private static final byte ZERO_BYTE = (byte)0;

	private double s;
	private double s2;
	private double tmp;
	private int count;

	private double maxDist = 999;

	private final double[] res;
	private long[] dims;
	private final long[] pos;

	private double[] vals;
	private boolean[] flags;

	private final int nd;

	private float[] distArr;
	private ArrayImg< FloatType, FloatArray > dist;

	private boolean[] processedArr;

	private BinaryHeap2D heap;
	private ArrayImg< NativeBoolType, BooleanArray > processed;

	private final RandomAccessibleInterval<B> mask;
	
	private final NativeBoolType falseBool;
	private final NativeBoolType trueBool;
	private final boolean[] nbrFlags;

	private RandomAccessibleInterval< UnsignedByteType > procNum;

	private RandomAccessible< NativeBoolType > valid;

	public DistanceInit( final RandomAccessibleInterval<B> mask, final double[] res )
	{
		this.mask= mask;
		this.res = res;

		this.nd = mask.numDimensions();
		pos = new long[ nd ];

		final int M = 2 * nd;
		vals = new double[ M ];
		flags = new boolean[ M ];
		
		falseBool = new NativeBoolType( false );
		trueBool = new NativeBoolType( true );
		nbrFlags = new boolean[ nd ];
	}

	public DistanceInit( final RandomAccessibleInterval<B> mask )
	{
		this( mask, DoubleStream.generate( () -> 1 ).limit( mask.numDimensions() ).toArray());
	}

	public void setMaxDist( final double maxDist )
	{
		this.maxDist = maxDist;
	}
	
	public RandomAccessibleInterval< FloatType > run()
	{
		init2d();
		fastMarching();
		return dist;
	}

	/**
	 * the Fast marching distance computation (!assumes a length-6 array with
	 * opposite coordinates stacked one after the other)
	 */
	public final double minDistance3D(final double[] val, final boolean[] flag) {
		// s = a + b +c; s2 = a*a + b*b +c*c
		s = 0;
		s2 = 0;
		count = 0;

		for (int n = 0; n < 6; n += 2) {
			if (flag[n] && flag[n + 1]) {
				tmp = min(val[n], val[n + 1]); // Take the smaller one if both are processed
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
		if( count == 0 )
			return maxDist;

		// count must be greater than zero since there must be at least one processed pt
		// in the neighbors
		tmp = (s + Math.sqrt((s * s - count * (s2 - 1.0f)))) / count;

		// The larger root
		return tmp;
	}

	public final <T extends RealType<T>, B extends BooleanType<B>> double minDistance3D(final RandomAccess<B> mask, final RandomAccess<T> access ) {
		int i = 0;
		for (int d = 0; d < 3; d++) {
			access.bck(d);
			mask.bck(d);
			vals[i] = access.get().getRealDouble();
			flags[i] = mask.get().get();
			i++;

			access.move(2, d);
			mask.move(2, d);
			vals[i] = access.get().getRealDouble();
			flags[i] = mask.get().get();
			access.bck(d);
			mask.bck(d);
			i++;
		}
		return minDistance3D( vals, flags );
	}

	
	/**
	 * the Fast marching distance computation (!assumes a length-6 array with
	 * opposite coordinates stacked one after the other)
	 */
	public final double minDistance2D(final double[] val, final boolean[] flag) {
		// s = a + b ; s2 = a*a + b*b 
		s = 0;
		s2 = 0;
		count = 0;

		for (int n = 0; n < 4; n += 2) {
			if (flag[n] && flag[n + 1]) {
				tmp = min(val[n], val[n + 1]); // Take the smaller one if both are processed
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
		if( count == 0 )
		{
//			System.out.println( "oh no" );
			return maxDist;
		}

		// count must be greater than zero since there must be at least one processed pt
		// in the neighbors
		tmp = (s + Math.sqrt((s * s - count * (s2 - 1.0f)))) / count;

		// The larger root
		return tmp;
	}

	public final <T extends RealType<T>, B extends BooleanType<B>, C extends BooleanType<C>> double minDistance2D(final RandomAccess<B> mask, final RandomAccess<T> dist, final RandomAccess<C> valid ) {
		
//		if( access.getIntPosition( 0 ) == 3 && access.getIntPosition( 1 ) == 2 )
//		{
//			System.out.println( "debug");
//		}
		
		int i = 0;
		for (int d = 0; d < 2; d++) {
			dist.bck( d );
			mask.bck( d );
			valid.bck( d );
			vals[i] = dist.get().getRealDouble();
			flags[i] = mask.get().get();
//			flags[i] = mask.get().get() && valid.get().get();
			i++;

			dist.move(2, d);
			mask.move(2, d);
			valid.move(2, d);
			vals[i] = dist.get().getRealDouble();
			flags[i] = mask.get().get();
//			flags[i] = mask.get().get() && valid.get().get();
			dist.bck(d);
			mask.bck(d);
			valid.bck( d );
			i++;
		}
		return minDistance2D( vals, flags );
	}
	
	/**
	 * the Fast marching distance computation (!assumes a length- 2*nd array with
	 * opposite coordinates stacked one after the other)
	 */
	public final double minDistance(final double[] val, final boolean[] flag) {
		// s = a + b ; s2 = a*a + b*b 
		s = 0;
		s2 = 0;
		count = 0;
		final int N = 2 * nd;
		for (int n = 0; n < N; n += 2) {
			if (flag[n] && flag[n + 1]) {
				tmp = min(val[n], val[n + 1]); // Take the smaller one if both are processed
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
		if( count == 0 )
		{
			System.out.println( "oh no" );
			return maxDist;
		}

		// count must be greater than zero since there must be at least one processed pt
		// in the neighbors
		tmp = (s + Math.sqrt((s * s - count * (s2 - 1.0f)))) / count;

		// The larger root
		return tmp;
	}
	
	public final <T extends RealType<T>, B extends BooleanType<B>, C extends BooleanType<C>> double minDistance(final RandomAccess<B> mask, final RandomAccess<T> dist, final RandomAccess<C> valid ) {

		int i = 0;
		for (int d = 0; d < nd; d++) {
			dist.bck( d );
			mask.bck( d );
			valid.bck( d );
			vals[i] = dist.get().getRealDouble();
			flags[i] = mask.get().get();
//			flags[i] = mask.get().get() && valid.get().get();
			i++;

			dist.move(2, d);
			mask.move(2, d);
			valid.move(2, d);
			vals[i] = dist.get().getRealDouble();
			flags[i] = mask.get().get();
//			flags[i] = mask.get().get() && valid.get().get();
			dist.bck(d);
			mask.bck(d);
			valid.bck( d );
			i++;
		}
		return minDistance2D( vals, flags );
	}

	public final double maskInit( final boolean[] flag )
	{
		if( nd == 2 )
			return maskInit2D( flag );
		else if ( nd == 3 )
			return maskInit3D( flag );
		else
			return -1;
	}
	
	public final double maskInit2D( final boolean[] flag )
	{
		final boolean fx = flag[ 0 ] || flag[ 1 ];
		final boolean fy = flag[ 2 ] || flag[ 3 ];

		if( fx )
		{
			if( fy )
			{
				// both
				final double x2 = res[0] * res[0];
				final double h = Math.sqrt( x2 + res[ 0 ] * res[ 0 ] );
				final double hx = h * res[ 1 ] / ( res[0] + res[ 1 ]);
//				final double d2 = x2 - ( hx * hx );
				return Math.sqrt( x2 - ( hx * hx ) );
			}
			else
			{
				return res[ 0 ] / 2;
			}
			
		}
		else if( fy )
		{
			return res[ 1 ] / 2;
		}
		else
		{
			return Double.NaN;
		}
	}
	
	public final double maskInit3D( final boolean[] flag )
	{
		final boolean fx = flag[ 0 ] || flag[ 1 ];
		final boolean fy = flag[ 2 ] || flag[ 3 ];
		final boolean fz = flag[ 4 ] || flag[ 5 ];

		if( fx )
		{
			if( fy )
			{
				// both
				final double x2 = res[0] * res[0];
				final double h = Math.sqrt( x2 + res[ 0 ] * res[ 0 ] );
				final double hx = h * res[ 1 ] / ( res[0] + res[ 1 ]);
//				final double d2 = x2 - ( hx * hx );
				return Math.sqrt( x2 - ( hx * hx ) );
			}
			else
			{
				return res[ 0 ] / 2;
			}
			
		}
		else if( fy )
		{
			return res[ 1 ] / 2;
		}
		else
		{
			return Double.NaN;
		}
	}
	
	public final <T extends RealType<T>, B extends BooleanType<B>> double maskInit2D( final RandomAccess<B> mask ) {
		int i = 0;
		for (int d = 0; d < 2; d++) {
			mask.bck(d);
			flags[i] = mask.get().get();
			i++;

			mask.move(2, d);
			flags[i] = mask.get().get();
			mask.bck(d);
			i++;
		}
//		System.out.println( "flags: " + Arrays.toString( flags ));

		return maskInit2D( flags );
	}

	public final <T extends RealType<T>, B extends BooleanType<B>> void init( RandomAccessibleInterval<B> mask, RandomAccessibleInterval<T> dist )
	{

		final RandomAccess< B > mra = Views.extendBorder( mask ).randomAccess();
		final RandomAccess< T > dra = Views.extendZero( dist ).randomAccess();


//		final byte b = ( byte ) 255;
//		final int nVoxels = (int)Intervals.numElements( mask );
//		heap = new BinaryHeap2D(nVoxels, BinaryHeap2D.MINTREE);	

		final Cursor< T > it = Views.flatIterable( dist ).cursor();
		int i = 0;
		while( it.hasNext() )
		{
			it.fwd();
			mra.setPosition( it );
			dra.setPosition( it );
			final double d = minDistance3D( mra, dra );
			it.get().setReal( d );

//			heap.addValue( (float)d, i, b );
			i++;
		}

	}
	
	
	public final void init2d()
	{
		final int N = (int)Intervals.numElements( mask );


		dims = Intervals.dimensionsAsLongArray( mask );
		distArr = new float[ N ];
		dist = ArrayImgs.floats( distArr, dims );

		processedArr = new boolean[ N ];
		processed = ArrayImgs.booleans( processedArr, dims );
		
		valid = Views.extendValue( 
					Views.interval( ConstantUtils.constantRandomAccessible( new NativeBoolType(true), nd ), mask ),
					falseBool );

		procNum = Converters.convertRAI( processed, (x,y) -> {
			if( x.get() )
				y.setOne();
			else
				y.setZero();
		}, new UnsignedByteType() );

		final RandomAccess< B > mra = Views.extendBorder( mask ).randomAccess();

		final OutOfBounds< NativeBoolType > pra = Views.extendValue( processed, falseBool ).randomAccess();
		final OutOfBounds< FloatType > dra = Views.extendValue( dist, Double.MAX_VALUE ).randomAccess();

		final int nVoxels = (int)Intervals.numElements( mask );
		heap = new BinaryHeap2D(nVoxels, BinaryHeap2D.MINTREE);	

		final Cursor< FloatType > c = Views.flatIterable( dist ).cursor();
		int i = -1;
		while( c.hasNext() )
		{
			c.fwd(); i++;
//			System.out.println( "c: " + Util.printCoordinates( c ));
			mra.setPosition( c );
			pra.setPosition( c );
			if( mra.get().get() )
			{
				pra.get().set( true );
				continue;
			}

//			final double d = minDistance2D( mra, dra );
			final double d = maskInit2D( mra );

			if( Double.isNaN( d ))
			{
				continue;
			}

//			System.out.println( String.format( "   added (%d) : %f", i, d ));
			heap.addValue( (float)d, i, ZERO_BYTE );

//			dra.setPosition( c );
//			pra.get().set( true );
	
//			addNeighbors2d( pra, dra );
		}
		
	}

	public <B extends BooleanType<B>, C extends BooleanType<C>, T extends RealType<T>> void addNeighbors2d( RandomAccess<B> processed, RandomAccess<T> dists, RandomAccess<C> valid )
	{
		long p = 0;
		double distance = 0;
		for (int d = 0; d < 2; d++) {
			dists.bck(d);
			processed.bck(d);
//			boolean proc = processed.get().get();
			if( !processed.get().get() && Intervals.contains( dist, dists ) )
			{
				p = IntervalIndexer.positionToIndexForInterval( dists, mask );
				distance = minDistance2D( processed, dists, valid );
//				System.out.println( "adding " + Util.printCoordinates( dists ) + distance );
				heap.addValue( (float)distance, (int)p, ZERO_BYTE );
			}

			dists.move(2, d);
			processed.move(2, d);
//			proc = processed.get().get();
			if( !processed.get().get() && Intervals.contains( dist, dists ) )
			{
				p = IntervalIndexer.positionToIndexForInterval( dists, mask );
				distance = minDistance2D( processed, dists, valid );
//				System.out.println( "adding " + Util.printCoordinates( dists ) + distance );
				heap.addValue( (float)distance, (int)p, ZERO_BYTE );
			}

			dists.bck(d);
			processed.bck(d);
		}
	}
	
	public <B extends BooleanType<B>, C extends BooleanType<C>, T extends RealType<T>> void addNeighbors( RandomAccess<B> processed, RandomAccess<T> dists, RandomAccess<C> valid )
	{
		long p = 0;
		double distance = 0;
		for (int d = 0; d < nd; d++) {
			dists.bck(d);
			processed.bck(d);
//			boolean proc = processed.get().get();
			if( !processed.get().get() && Intervals.contains( dist, dists ) )
			{
				p = IntervalIndexer.positionToIndexForInterval( dists, mask );
				distance = minDistance( processed, dists, valid );
//				System.out.println( "adding " + Util.printCoordinates( dists ) + distance );
				heap.addValue( (float)distance, (int)p, ZERO_BYTE );
			}

			dists.move(2, d);
			processed.move(2, d);
//			proc = processed.get().get();
			if( !processed.get().get() && Intervals.contains( dist, dists ) )
			{
				p = IntervalIndexer.positionToIndexForInterval( dists, mask );
				distance = minDistance( processed, dists, valid );
//				System.out.println( "adding " + Util.printCoordinates( dists ) + distance );
				heap.addValue( (float)distance, (int)p, ZERO_BYTE );
			}

			dists.bck(d);
			processed.bck(d);
		}
	}

	public static final double min(double a, double b) {
		if (a < b)
			return a;
		else
			return b;
	}
	
	public <T extends RealType<T>> void fastMarching()
	{
		final OutOfBounds< NativeBoolType > pra = Views.extendValue( processed, falseBool ).randomAccess();
		final OutOfBounds< FloatType > dra = Views.extendValue( dist , maxDist).randomAccess();
		final RandomAccess< NativeBoolType > vra = valid.randomAccess();
//		final RandomAccess< B > mra = mask.randomAccess();

		while( heap.isNotEmpty() )
		{
			float v = heap.getFirst();
			int posIdx = heap.getFirstId();
			byte s = heap.getFirstState();	
			heap.removeFirst();

			if( processedArr[ posIdx ])
			{
				continue;
			}
			
//			System.out.println( "setting " + Arrays.toString( pos ) + " : " + v );
			processedArr[ posIdx ] = true;
			distArr[ posIdx ] = v;
			
//			printStatus();

			IntervalIndexer.indexToPosition( posIdx, dims, pos );
			dra.setPosition( pos );
			pra.setPosition( pos );
			
			addNeighbors( pra, dra, vra );
		}
	}
	
	public void emptyAndPrintHeap()
	{
		while( heap.isNotEmpty() )
		{
			float v = heap.getFirst();
			int xy = heap.getFirstId();
			byte s = heap.getFirstState();	
			heap.removeFirst();
			IntervalIndexer.indexToPosition( xy, dims, pos );
			
//			System.out.println( Arrays.toString( pos ) + " : " + v );

		}
		
	}

	public void printStatus()
	{
		System.out.println( "" );
		printImg2d( procNum );
		System.out.println( "" );
		printImg2d( dist );
		System.out.println( "" );	
	}

	public static <T extends NumericType<T>> void printImg2d( final RandomAccessibleInterval<T> img )
	{
		Cursor< T > c = Views.flatIterable( img ).cursor();
		int y = 0;
		while( c.hasNext() )
		{
			c.fwd();
			if( c.getIntPosition( 1 ) != y )
			{
				System.out.print( "\n" );
				y = c.getIntPosition( 1 );
			}
			
			System.out.print( c.get() + " ");
		}
		System.out.print( "\n" );

	}

	public static void main( String[] args )
	{
		double[] res = new double[] { 1, 1 };
		
		boolean[] flags = new boolean[] { true, false, true, false };
		double[] dists = new double[] { 1, -1, 1, -1 };

//		ArrayImg< BitType, LongArray > mask = ArrayImgs.bits( 4, 4 );
//		ArrayRandomAccess< BitType > maskAccess = mask.randomAccess();
//		maskAccess.setPosition( 1, 0 ); maskAccess.get().set( true );
//		maskAccess.fwd( 1 ); maskAccess.get().set( true );
//		maskAccess.fwd( 0 ); maskAccess.get().set( true );
//		
		
		final boolean[] maskData = new boolean[] {
			false,  true,  true,  true,  true,
			false, false,  true,  true,  true,
			false,  true,  true,  true,  true,
			false,  true,  true,  true, false,
			false,  true,  true, false, false
		};
		ArrayImg< NativeBoolType, BooleanArray > mask = ArrayImgs.booleans( maskData, 5, 5 );

		RandomAccessibleInterval< NativeBoolType > maskInv = Converters.convertRAI( mask, ( x, y ) -> { y.set( !x.get() ); }, new NativeBoolType() );

//		Views.interval( mask, Intervals.createMinMax( 0, 0, 3, 1 ) ).forEach( x -> x.set( true ) );
//		mask.getAt( 1,1 ).set( true );


//		DistanceInit di = new DistanceInit( mask, res );
//		double d = di.minDistance( dists, flags );
//		System.out.println( d );
		
		
//		printImg2d( mask );
//		System.out.println(" ");
//
//		di.init2d();
////		printImg2d( di.dist );
//		System.out.println("");
//		
////		di.emptyAndPrintHeap();
//
//		di.fastMarching();
//		printImg2d( di.dist );
		
		
		DistanceInit di = new DistanceInit( maskInv, res );
		RandomAccessibleInterval df = di.run();
		DistanceInit.printImg2d( df );

		System.out.println("");
		System.out.println("done");
	}
	

}
