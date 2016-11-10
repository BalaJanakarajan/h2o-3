package water.fvec;

import org.junit.*;

import water.Futures;
import water.TestUtil;
import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.assertTrue;

public class C8ChunkTest extends TestUtil {
  @BeforeClass() public static void setup() { stall_till_cloudsize(1); }
  @Test
  public void test_inflate_impl() {
    for (int l=0; l<2; ++l) {
      NewChunk nc = new NewChunk(Vec.T_NUM);

      long[] vals = new long[]{Long.MIN_VALUE+1, Integer.MIN_VALUE, 0, Integer.MAX_VALUE, Long.MAX_VALUE};
      if (l==1) nc.addNA();
      for (long v : vals) nc.addNum(v, 0);
      nc.addNA(); //-9223372036854775808l

      Chunk cc = nc.compress();
      Assert.assertEquals(vals.length + 1 + l, cc._len);
      assertTrue(cc instanceof C8Chunk);
      for (int i = 0; i < vals.length; ++i) Assert.assertEquals(vals[i], cc.at8(l + i));
      assertTrue(cc.isNA(vals.length + l));
      double[] densevals = new double[cc.len()];
      cc.getDoubles(densevals,0,cc.len());
      for (int i = 0; i < densevals.length; ++i) {
        if (cc.isNA(i)) assertTrue(Double.isNaN(densevals[i]));
        else assertTrue(cc.at8(i)==densevals[i]);
      }

      nc = new NewChunk(Vec.T_NUM);
      cc.inflate_impl(nc);
      nc.values(0, nc._len);
      if (l==1) assertTrue(cc.isNA(0));
      Assert.assertEquals(vals.length+l+1, nc._sparseLen);
      Assert.assertEquals(vals.length+l+1, nc._len);
      Iterator<NewChunk.Value> it = nc.values(0, vals.length+1+l);
      for (int i = 0; i < vals.length+1+l; ++i) assertTrue(it.next().rowId0() == i);
      assertTrue(!it.hasNext());
      for (int i = 0; i < vals.length; ++i) Assert.assertEquals(vals[i], nc.at8(l + i));
      assertTrue(cc.isNA(vals.length + l));

      Chunk cc2 = nc.compress();
      Assert.assertEquals(vals.length + 1 + l, cc._len);
      assertTrue(cc2 instanceof C8Chunk);
      for (int i = 0; i < vals.length; ++i) Assert.assertEquals(vals[i], cc2.at8(l + i));
      assertTrue(cc2.isNA(vals.length + l));

      assertTrue(Arrays.equals(cc._mem, cc2._mem));
    }
  }

  @Test public void test_setNA() {
    // Create a vec with one chunk with 15 elements, and set its numbers
    water.Key key = Vec.newKey();
    Vec vec = new Vec(key, Vec.ESPC.rowLayout(key,new long[]{0,15}),1).makeZero();
    long[] vals = new long[]{Long.MIN_VALUE+1, 1, 0, 2, 0, 51, 0, 33, 0, 21234, 3422, 3767, 0, 0, Long.MAX_VALUE};
    Vec.Writer w = vec.open();
    for (int i =0; i<vals.length; ++i) w.set(i, vals[i]);
    w.close();

    ChunkAry cc = vec.chunkForChunkIdx(0);
    assertTrue(cc.getChunk(0) instanceof C8Chunk);
    Futures fs = new Futures();
    fs.blockForPending();

    for (int i = 0; i < vals.length; ++i) Assert.assertEquals(vals[i], cc.at8(i), Double.MIN_VALUE);

    int[] NAs = new int[]{1, 5, 2};
    int[] notNAs = new int[]{0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    w = vec.open();
    for (int i:NAs) w.setNA(i);
    w.close();

    for (int na : NAs) assertTrue(cc.isNA(na));
    for (int notna : notNAs) assertTrue(!cc.isNA(notna));

    NewChunk nc = cc.getChunkInflated(0);
    nc.values(0, nc._len);
    Assert.assertEquals(vals.length, nc._sparseLen);
    Assert.assertEquals(vals.length, nc._len);

    Iterator<NewChunk.Value> it = nc.values(0, vals.length);
    for (int i = 0; i < vals.length; ++i) assertTrue(it.next().rowId0() == i);
    assertTrue(!it.hasNext());

    for (int na : NAs) assertTrue(cc.isNA(na));
    for (int notna : notNAs) assertTrue(!cc.isNA(notna));

    Chunk cc2 = nc.compress();
    Assert.assertEquals(vals.length, cc._len);
    assertTrue(cc2 instanceof C8Chunk);
    for (int na : NAs) assertTrue(cc.isNA(na));
    for (int notna : notNAs) assertTrue(!cc.isNA(notna));

    assertTrue(Arrays.equals(cc.getChunk(0)._mem, cc2._mem));
    vec.remove();
  }
}
