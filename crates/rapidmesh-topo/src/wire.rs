//! A flat, POD wire format for the complexes. Every topology field is a
//! `Vec<[T; N]>` of plain-old-data, so encoding is just length-prefixed byte
//! blocks (no per-element serialization) and decoding is a bulk copy. The
//! resulting buffer is self-describing and mmap-friendly for the cross-process
//! path; geometry is intentionally *not* framed — it is recomputed from `coords`
//! in one O(n) pass, far cheaper than shipping it.

use crate::{Csr, TetTopology, TriTopology};
use bytemuck::Pod;

const MAGIC: u32 = 0x5254_4f50; // "RTOP"
const VERSION: u32 = 1;

/// Appends POD slices as length-prefixed blocks behind a small header.
pub struct FrameWriter {
    buf: Vec<u8>,
    n: u32,
}

impl Default for FrameWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameWriter {
    pub fn new() -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // block count, patched in finish()
        FrameWriter { buf, n: 0 }
    }

    /// Append one POD block.
    pub fn push<T: Pod>(&mut self, s: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(s);
        self.buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(bytes);
        self.n += 1;
    }

    /// Finish, returning the framed buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.buf[8..12].copy_from_slice(&self.n.to_le_bytes());
        self.buf
    }
}

/// Reads the blocks back, in the order they were written.
pub struct FrameReader<'a> {
    buf: &'a [u8],
    pos: usize,
    remaining: u32,
}

impl<'a> FrameReader<'a> {
    pub fn new(buf: &'a [u8]) -> Option<Self> {
        if buf.len() < 12 {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        let version = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        if magic != MAGIC || version != VERSION {
            return None;
        }
        let remaining = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        Some(FrameReader { buf, pos: 12, remaining })
    }

    /// Read the next block as `Vec<T>` (bulk copy; tolerates misalignment).
    pub fn next<T: Pod>(&mut self) -> Option<Vec<T>> {
        if self.remaining == 0 || self.pos + 8 > self.buf.len() {
            return None;
        }
        let len = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().ok()?) as usize;
        self.pos += 8;
        if self.pos + len > self.buf.len() {
            return None;
        }
        let sz = core::mem::size_of::<T>();
        if sz == 0 || len % sz != 0 {
            return None;
        }
        let bytes = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        self.remaining -= 1;
        Some(bytemuck::pod_collect_to_vec(bytes))
    }
}

impl TriTopology {
    /// Encode to the flat POD wire format.
    pub fn to_wire(&self) -> Vec<u8> {
        let mut w = FrameWriter::new();
        w.push(&[self.n_verts as u64]);
        w.push(&self.tris);
        w.push(&self.edges);
        w.push(&self.tri_edges);
        w.push(&self.edge_tris);
        w.push(&self.edge_tags);
        let (o, d) = self.vert_tris.as_raw();
        w.push(o);
        w.push(d);
        w.finish()
    }

    /// Decode from the wire format. `None` on a malformed buffer.
    pub fn from_wire(buf: &[u8]) -> Option<Self> {
        let mut r = FrameReader::new(buf)?;
        let n_verts = *r.next::<u64>()?.first()? as usize;
        let tris = r.next()?;
        let edges = r.next()?;
        let tri_edges = r.next()?;
        let edge_tris = r.next()?;
        let edge_tags = r.next()?;
        let vt_o = r.next()?;
        let vt_d = r.next()?;
        Some(TriTopology {
            n_verts,
            tris,
            edges,
            tri_edges,
            edge_tris,
            edge_tags,
            vert_tris: Csr::from_raw(vt_o, vt_d),
        })
    }
}

impl TetTopology {
    /// Encode to the flat POD wire format.
    pub fn to_wire(&self) -> Vec<u8> {
        let mut w = FrameWriter::new();
        w.push(&[self.n_verts as u64]);
        w.push(&self.tets);
        w.push(&self.edges);
        w.push(&self.faces);
        w.push(&self.tet_edges);
        w.push(&self.tet_edge_sign);
        w.push(&self.tet_faces);
        w.push(&self.tet_face_sign);
        w.push(&self.face_edges);
        w.push(&self.face_tets);
        let (veo, ved) = self.vert_edges.as_raw();
        w.push(veo);
        w.push(ved);
        let (vto, vtd) = self.vert_tets.as_raw();
        w.push(vto);
        w.push(vtd);
        w.finish()
    }

    /// Decode from the wire format. `None` on a malformed buffer.
    pub fn from_wire(buf: &[u8]) -> Option<Self> {
        let mut r = FrameReader::new(buf)?;
        let n_verts = *r.next::<u64>()?.first()? as usize;
        let tets = r.next()?;
        let edges = r.next()?;
        let faces = r.next()?;
        let tet_edges = r.next()?;
        let tet_edge_sign = r.next()?;
        let tet_faces = r.next()?;
        let tet_face_sign = r.next()?;
        let face_edges = r.next()?;
        let face_tets = r.next()?;
        let veo = r.next()?;
        let ved = r.next()?;
        let vto = r.next()?;
        let vtd = r.next()?;
        Some(TetTopology {
            n_verts,
            tets,
            edges,
            faces,
            tet_edges,
            tet_edge_sign,
            tet_faces,
            tet_face_sign,
            face_edges,
            face_tets,
            vert_edges: Csr::from_raw(veo, ved),
            vert_tets: Csr::from_raw(vto, vtd),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tets, Tris};
    use crate::{TetTopology, TriTopology};

    #[test]
    fn tri_roundtrip() {
        let topo = TriTopology::build(&Tris {
            tris: &[[0, 1, 2], [1, 3, 2]],
            tags: &[7, 9],
            n_verts: 4,
        });
        let back = TriTopology::from_wire(&topo.to_wire()).unwrap();
        assert_eq!(topo, back);
    }

    #[test]
    fn tet_roundtrip() {
        let topo = TetTopology::build(&Tets { tets: &[[0, 1, 2, 3], [1, 2, 3, 4]], n_verts: 5 });
        let back = TetTopology::from_wire(&topo.to_wire()).unwrap();
        assert_eq!(topo, back);
    }

    #[test]
    fn rejects_garbage() {
        assert!(TetTopology::from_wire(&[0u8; 4]).is_none());
        assert!(TetTopology::from_wire(b"not a frame!").is_none());
    }
}
