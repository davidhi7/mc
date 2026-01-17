use std::{collections::HashMap, sync::Arc};

use crate::{
    math::nd_array::ShiftedHyperCubeArray,
    world::{
        self,
        blocks::{Block, BlockPhysicsType},
        chunk::{CHUNK_WIDTH, CHUNK_WIDTH_I32, Chunk, ChunkStack, ChunkUVW, ChunkUW},
        world_gen::ChunkGenResult,
    },
};
use glam::{IVec3, USizeVec3};
use itertools::Itertools;

pub mod blocks;
pub mod chunk;
pub mod world_gen;
pub mod world_loader;

pub trait LookupBlock {
    fn lookup_block(&self, block: IVec3) -> Option<Block>;

    fn is_solid(&self, block: IVec3) -> bool {
        self.lookup_block(block)
            .is_some_and(|block| matches!(block.physics_type(), BlockPhysicsType::Solid))
    }

    #[expect(dead_code)]
    fn is_liquid(&self, block: IVec3) -> bool {
        self.lookup_block(block)
            .is_some_and(|block| matches!(block.physics_type(), BlockPhysicsType::Liquid))
    }
}

#[derive(Default)]
enum TriState<T> {
    #[default]
    Missing,
    Present(Option<T>),
}

impl<T> TriState<T> {
    fn is_missing(&self) -> bool {
        matches!(self, TriState::Missing)
    }

    fn is_present(&self) -> bool {
        matches!(self, TriState::Present(_))
    }

    fn present_from_option(option: Option<T>) -> Self {
        Self::Present(option)
    }
}

struct AdjacentState {
    state: ShiftedHyperCubeArray<2, 3, TriState<Vec<(IVec3, Block)>>>,
}

impl AdjacentState {
    fn new() -> Self {
        Self {
            state: ShiftedHyperCubeArray::default([-1; 2]),
        }
    }

    fn all_present(&self) -> bool {
        self.state.iter().enumerate().all(|(idx, vec)| {
            // Check if each item is in the center or adjacent and present
            idx == 9 / 2 || vec.is_present()
        })
    }

    fn insert_adjacent_state(&mut self, index: [isize; 2], state: Option<Vec<(IVec3, Block)>>) {
        assert!(self.state[index].is_missing(), "Adjacent state already set");
        self.state[index] = TriState::present_from_option(state);
    }

    fn complete_chunk_stack(&self, chunk_stack: &ChunkStack) {
        for (pos, block) in self
            .state
            .iter()
            .enumerate()
            .filter_map(|(i, vec)| {
                if i == 9 / 2 {
                    // Ignore center element
                    return None;
                }
                match vec {
                    TriState::Missing => panic!("All adjacent chunk stacks should be completed"),
                    TriState::Present(optional_vec) => optional_vec.as_ref(),
                }
            })
            .flatten()
        {
            // TODO prevent asserts on type level
            let (uvw, inner) = world::divide_world_coordinates(*pos);
            assert_eq!(uvw.u, 0);
            assert_eq!(uvw.w, 0);
            chunk_stack
                .get_chunk(uvw.v)
                .expect("Invalid chunk v coordinate")
                .set(inner, *block);
        }
    }
}

enum ChunkStackState {
    Missing(AdjacentState),
    Incomplete(Arc<ChunkStack>, AdjacentState),
    Complete(Arc<ChunkStack>),
}

impl ChunkStackState {
    fn empty() -> Self {
        ChunkStackState::Missing(AdjacentState::new())
    }

    fn from_chunk_stack(chunk_stack: Arc<ChunkStack>) -> Self {
        ChunkStackState::Incomplete(chunk_stack, AdjacentState::new())
    }

    fn with_chunk_stack(self, chunk_stack: Arc<ChunkStack>) -> Self {
        match self {
            ChunkStackState::Missing(neighbors_state) => {
                if neighbors_state.all_present() {
                    neighbors_state.complete_chunk_stack(&chunk_stack);
                    ChunkStackState::Complete(chunk_stack)
                } else {
                    ChunkStackState::Incomplete(chunk_stack, neighbors_state)
                }
            }

            ChunkStackState::Incomplete(..) | ChunkStackState::Complete(..) => panic!(
                "Ilegal state: Chunk stack at {:?} already loaded",
                chunk_stack.uw()
            ),
        }
    }
}

pub struct World {
    chunk_stacks: HashMap<ChunkUW, ChunkStackState>,
}

impl World {
    pub fn new() -> Self {
        World {
            chunk_stacks: HashMap::new(),
        }
    }

    /// Returns the chunk stack if generated (complete or incomplete), otherwise None.
    pub fn get_chunk_stack(&self, uw: ChunkUW) -> Option<&Arc<ChunkStack>> {
        match self.chunk_stacks.get(&uw)? {
            ChunkStackState::Complete(chunk_stack)
            | ChunkStackState::Incomplete(chunk_stack, _) => Some(chunk_stack),
            ChunkStackState::Missing(_) => None,
        }
    }

    /// Returns the chunk if completely generated, otherwise None.
    pub fn get_chunk(&self, uvw: ChunkUVW) -> Option<&Chunk> {
        self.get_chunk_stack(uvw.to_uw())?.get_chunk(uvw.v)
    }

    /// Returns boolean indicating whether this chunk stack is complete,
    /// e.g. all adjacent chunk stacks were generated and generation of this chunk stack is completed.
    /// Returns None, if this chunk stack is not present yet.
    pub fn is_complete(&self, uw: ChunkUW) -> bool {
        matches!(
            self.chunk_stacks.get(&uw),
            Some(ChunkStackState::Complete(_))
        )
    }

    /// Returns true if chunk stack is generated, regardless of whether it is completed or incomplete.
    pub fn is_generated(&self, uw: ChunkUW) -> bool {
        self.chunk_stacks.get(&uw).is_some_and(|chunk_stack| {
            matches!(
                chunk_stack,
                ChunkStackState::Incomplete(..) | ChunkStackState::Complete(..)
            )
        })
    }

    /// Insert new chunk stack into the world. This panics if a chunk stack with the same UW coordinates was inserted before.
    /// Returns all UW coordinates for chunk stacks that are completed by this chunk stack insertion.
    pub fn insert_chunk_stack(&mut self, chunk_gen: ChunkGenResult) -> Vec<ChunkUW> {
        let mut completed_chunks = Vec::new();

        let ChunkGenResult {
            chunk_stack,
            adjacent_chunk_changes,
        } = chunk_gen;
        let uw = chunk_stack.uw();

        let new_state = match self.chunk_stacks.remove(&uw) {
            Some(state) => state.with_chunk_stack(chunk_stack),
            None => ChunkStackState::from_chunk_stack(chunk_stack),
        };
        if let ChunkStackState::Complete(_) = new_state {
            completed_chunks.push(uw);
        }

        self.chunk_stacks.insert(uw, new_state);

        // Notify adjacent chunk stacks of this chunk stack's insertion
        for ([u_offset, w_offset], buffer) in adjacent_chunk_changes.into_iter_enumerated() {
            // Ignore center chunk, only consider truly adjacent ones
            if u_offset == 0 && w_offset == 0 {
                continue;
            }

            // TODO prettier arithmetics
            let adjacent_uw = ChunkUW {
                u: uw.u + u_offset as i32,
                w: uw.w + w_offset as i32,
            };

            let mut state = self
                .chunk_stacks
                .remove(&adjacent_uw)
                .unwrap_or(ChunkStackState::empty());

            match &mut state {
                ChunkStackState::Missing(neighbors_state)
                | ChunkStackState::Incomplete(_, neighbors_state) => {
                    neighbors_state.insert_adjacent_state([-u_offset, -w_offset], buffer);
                }
                ChunkStackState::Complete(_) => {
                    panic!("Illegal state: Adjacent chunk stack shouldn't be complete")
                }
            }

            // Check if the adjacent chunk stack's state is complete
            if let ChunkStackState::Incomplete(chunk_stack, neighbors_state) = &state
                && neighbors_state.all_present()
            {
                neighbors_state.complete_chunk_stack(chunk_stack);
                completed_chunks.push(adjacent_uw);

                state = ChunkStackState::Complete(Arc::clone(chunk_stack));
            }
            self.chunk_stacks.insert(adjacent_uw, state);
        }

        completed_chunks
    }

    /// Replace the block at the given coordinates.
    /// This function returns all chunks that need to be reloaded.
    /// This can be more than one chunk, for example if the replaced block is adjacent to blocks in another chunk and their visibility or AO values change.
    pub fn replace_block(&mut self, coords: IVec3, block: Block) -> Vec<ChunkUVW> {
        let (uvw, inner_chunk_coords) = divide_world_coordinates(coords);
        if !ChunkStack::validate_chunk_v(uvw.v) {
            return Vec::new();
        }

        let Some(chunk) = self.get_chunk(uvw) else {
            return Vec::new();
        };
        chunk.set(inner_chunk_coords, block);

        // TODO better check if remeshing is required
        let mut updated_chunks = vec![];

        // Find all chunks sharing a block face with the replaced block.
        // These neighboring chunks are represented by the vector offset relative to the origin chunk's uvw coordinates.
        // Using the powerset of this set later, we also represent chunks that only share an edge or single point.
        // At most three elements are in this set (unit vectors of 3d vector space), so use array with counter instead of vec.
        let mut count = 0;
        let mut adjacent_to_chunks = [IVec3::ZERO; 3];
        if inner_chunk_coords.x == 0 {
            adjacent_to_chunks[count] = IVec3::NEG_X;
            count += 1;
        } else if inner_chunk_coords.x == CHUNK_WIDTH - 1 {
            adjacent_to_chunks[count] = IVec3::X;
            count += 1;
        };

        if inner_chunk_coords.y == 0 {
            adjacent_to_chunks[count] = IVec3::NEG_Y;
            count += 1;
        } else if inner_chunk_coords.y == CHUNK_WIDTH - 1 {
            adjacent_to_chunks[count] = IVec3::Y;
            count += 1;
        };

        if inner_chunk_coords.z == 0 {
            adjacent_to_chunks[count] = IVec3::NEG_Z;
            count += 1;
        } else if inner_chunk_coords.z == CHUNK_WIDTH - 1 {
            adjacent_to_chunks[count] = IVec3::Z;
            count += 1;
        };

        // The power set always contains the empty set, so this loop also always considers the empty directions vector
        for directions in adjacent_to_chunks.into_iter().take(count).powerset() {
            let chunk_offset: IVec3 = directions.iter().sum();
            let uvw: ChunkUVW = (IVec3::from(uvw) + chunk_offset).into();

            if self.get_chunk(uvw).is_some() {
                updated_chunks.push(uvw);
            }
        }

        updated_chunks
    }

    /// Clear all data associated with this world.
    pub fn clear(&mut self) {
        self.chunk_stacks.clear();
    }
}

impl LookupBlock for World {
    fn lookup_block(&self, coords: IVec3) -> Option<Block> {
        let (uvw, inner) = divide_world_coordinates(coords);
        self.get_chunk(uvw).map(|chunk| chunk.get(inner))
    }
}

pub fn divide_world_coordinates(pos: IVec3) -> (ChunkUVW, USizeVec3) {
    (
        pos.div_euclid(IVec3::splat(CHUNK_WIDTH_I32)).into(),
        pos.rem_euclid(IVec3::splat(CHUNK_WIDTH_I32)).as_usizevec3(),
    )
}
