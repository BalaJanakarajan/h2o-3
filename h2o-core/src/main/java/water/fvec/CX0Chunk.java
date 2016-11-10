package water.fvec;

import water.H2O;
import water.util.UnsafeUtils;

import java.util.Iterator;

/** specialized subtype of SPARSE chunk for boolean (bitvector); no NAs.  contains just a list of rows that are non-zero. */
public final class CX0Chunk extends CXIChunk {
  // Sparse constructor
  protected CX0Chunk(int len, byte [] buf){super(len,0,buf);}

  @Override public final long at8(int idx) {return getId(findOffset(idx)) == idx?1:0;}
  @Override public final double atd(int idx) { return at8(idx); }
  @Override public final boolean isNA( int i ) { return false; }
  @Override double min() { return 0; }
  @Override double max() { return 1; }
  @Override public boolean hasNA() { return false; }

  @Override public int asSparseDoubles(double [] vals, int[] ids, double NA) {
    if(vals.length < _sparseLen) throw new IllegalArgumentException();
    int off = _OFF;
    final int inc = _ridsz;
    if(_ridsz == 2){
      for (int i = 0; i < _sparseLen; ++i, off += inc) {
        ids[i] = UnsafeUtils.get2(_mem,off) & 0xFFFF;
        vals[i] = 1;
      }
    } else if(_ridsz == 4){
      for (int i = 0; i < _sparseLen; ++i, off += inc) {
        ids[i] = UnsafeUtils.get4(_mem,off);
        vals[i] = 1;
      }
    } else throw H2O.unimpl();
    return sparseLenZero();
  }

  public Iterator<Value> values(){
    return new SparseIterator(new Value(){
      @Override public final long asLong(){return 1;}
      @Override public final double asDouble() { return 1;}
      @Override public final boolean isNA(){ return false;}
    });
  }
}
