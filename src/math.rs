use glam::{IVec2, IVec3, Vec3};

pub mod nd_array;
pub mod ray_caster;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aabb2I {
    pub min: IVec2,
    pub max: IVec2,
    _private: (),
}

#[allow(dead_code)]
impl Aabb2I {
    pub fn new(min: IVec2, max: IVec2) -> Self {
        debug_assert!(!min.cmpgt(max).any());
        Self {
            min,
            max,
            _private: (),
        }
    }

    /// Check whether this AABB contains the point. Also returns true if the point is on the edge of the AABB.
    pub fn contains_point(&self, point: IVec2) -> bool {
        (self.min.x..=self.max.x).contains(&point.x) && (self.min.y..=self.max.y).contains(&point.y)
    }

    /// Check whether two AABB's intersect. Sharing a point or edge
    pub fn intersects(&self, other: Aabb2I) -> bool {
        (self.min.x < other.max.x && self.max.x > other.min.x)
            && (self.min.y < other.max.y && self.max.y > other.min.y)
    }
}

macro_rules! define_aabb_3d {
    ($type:ty, $name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            pub min: $type,
            pub max: $type,
            _private: (),
        }

        #[allow(dead_code)]
        impl $name {
            pub fn new(min: $type, max: $type) -> Self {
                debug_assert!(max.cmpge(min).all());
                Self {
                    min,
                    max,
                    _private: (),
                }
            }

            /// Check whether this AABB contains the point. Also returns true if the point is on the edge of the AABB.
            pub fn contains_point(&self, point: $type) -> bool {
                (self.min.x..=self.max.x).contains(&point.x)
                    && (self.min.y..=self.max.y).contains(&point.y)
                    && (self.min.z..=self.max.z).contains(&point.z)
            }

            /// Check whether two AABB's intersect. Sharing a point, edge or plane doens't count as intersection.
            pub fn intersects(&self, other: $name) -> bool {
                (self.min.x < other.max.x && self.max.x > other.min.x)
                    && (self.min.y < other.max.y && self.max.y > other.min.y)
                    && (self.min.z < other.max.z && self.max.z > other.min.z)
            }
        }
    };
}

define_aabb_3d!(Vec3, Aabb3);
define_aabb_3d!(IVec3, Aabb3I);

impl Aabb3 {
    /// Returns the smallest integer-based AABB that fully contains the the given AABB.
    /// This is done by calling `floor` on the min and `ceil` on the max value.
    pub fn to_ivec_aabb(self) -> Aabb3I {
        Aabb3I::new(self.min.floor().as_ivec3(), self.max.ceil().as_ivec3())
    }
}

#[allow(dead_code)]
impl Aabb3I {
    pub fn to_vec_aabb(self) -> Aabb3 {
        Aabb3::new(self.min.as_vec3(), self.max.as_vec3())
    }
}

#[cfg(test)]
mod tests {
    use glam::{ivec3, vec3};

    use super::*;

    #[test]
    fn test_to_ivec_aabb() {
        assert_eq!(
            Aabb3::new(vec3(0.0, 0.5, 1.5), vec3(1.0, 0.5, 2.5)).to_ivec_aabb(),
            Aabb3I::new(ivec3(0, 0, 1), ivec3(1, 1, 3))
        );
    }
}
