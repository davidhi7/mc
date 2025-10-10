use glam::{IVec2, IVec3, Vec3, ivec2, ivec3};

pub mod ray_caster;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aabb2I {
    pub min: IVec2,
    pub max: IVec2,
}

impl Aabb2I {
    pub fn new(min: IVec2, max: IVec2) -> Self {
        debug_assert!(!min.cmpgt(max).any());
        Self { min, max }
    }

    pub fn contains_point(&self, vec: IVec2) -> bool {
        (self.min.x..=self.max.x).contains(&vec.x) && (self.min.y..=self.max.y).contains(&vec.y)
    }
}

macro_rules! define_aabb_3d {
    ($type:ty, $name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            pub min: $type,
            pub max: $type,
        }

        impl $name {
            pub fn new(min: $type, max: $type) -> Self {
                debug_assert!(max.cmpge(min).all());
                Self { min, max }
            }

            pub fn contains_point(&self, vec: $type) -> bool {
                (self.min.x..=self.max.x).contains(&vec.x)
                    && (self.min.y..=self.max.y).contains(&vec.y)
                    && (self.min.z..=self.max.z).contains(&vec.z)
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
    pub fn to_ivec_aabb(&self) -> Aabb3I {
        Aabb3I {
            min: self.min.floor().as_ivec3(),
            max: self.max.ceil().as_ivec3(),
        }
    }
}

impl Aabb3I {
    pub fn to_vec_aabb(&self) -> Aabb3 {
        Aabb3 {
            min: self.min.as_vec3(),
            max: self.max.as_vec3(),
        }
    }
}

/// Returns a set of disjoint 2d areas within the first AABB, that aren't overlapped by the second AABB.
/// All returned areas don't share any edge with eachother or the subtracted area.
///
/// If the second AABB isn't fully covering the first AABB, and either:
///
/// 1. AABB don't overlap at all
/// 2. Min vectors aren't lower than max vectors in every dimension
/// 3. Either AABB has one side of zero length
///
/// Then `vec![(area_min, area_max)]` is returned.
pub fn area_subtract_overlap_2d(area: Aabb2I, subtracted_area: Aabb2I) -> Vec<Aabb2I> {
    let Aabb2I {
        min: area_min,
        max: area_max,
    } = area;

    let Aabb2I {
        min: subtracted_area_min,
        max: subtracted_area_max,
    } = subtracted_area;

    // Early return if the area is entirely within the subtracted area
    if subtracted_area_max.cmpge(area_max).all() && subtracted_area_min.cmple(area_min).all() {
        return vec![];
    }

    let overlap_min = area_min.max(subtracted_area_min);
    let overlap_max = area_max.min(subtracted_area_max);

    // If overlap area is degenerated, return full area of first AABB
    if (overlap_max - overlap_min).min_element() <= 0 {
        return vec![Aabb2I::new(area_min, area_max)];
    }

    let mut result = Vec::with_capacity(4);

    // negative x area
    if area_min.x < overlap_min.x {
        result.push(Aabb2I::new(area_min, ivec2(overlap_min.x - 1, area_max.y)));
    }

    // positive x area
    if area_max.x > overlap_max.x {
        result.push(Aabb2I::new(ivec2(overlap_max.x + 1, area_min.y), area_max));
    }

    // negative y area
    if area_min.y < overlap_min.y {
        result.push(Aabb2I::new(
            ivec2(overlap_min.x, area_min.y),
            ivec2(overlap_max.x, overlap_min.y - 1),
        ));
    }

    // positive y area
    if area_max.y > overlap_max.y {
        result.push(Aabb2I::new(
            ivec2(overlap_min.x, overlap_max.y + 1),
            ivec2(overlap_max.x, area_max.y),
        ));
    }

    result
}

/// Returns a set of disjoint 3d volumes within the first AABB, that aren't overlapped by the second AABB.
/// All returned volumes don't share any edge with eachother or the subtracted volume.
///
/// If the second AABB isn't fully covering the first AABB, and either:
///
/// 1. AABB don't overlap at all
/// 2. Min vectors aren't lower than max vectors in every dimension
/// 3. Either AABB has one side of zero length
///
/// Then `vec![(volume_min, volume_max)]` is returned.
pub fn volume_subtract_overlap_3d(volume: Aabb3I, subtracted_volume: Aabb3I) -> Vec<Aabb3I> {
    let Aabb3I {
        min: volume_min,
        max: volume_max,
    } = volume;

    let Aabb3I {
        min: subtracted_volume_min,
        max: subtracted_volume_max,
    } = subtracted_volume;

    // Early return if the area is entirely within the subtracted area
    if subtracted_volume_max.cmpge(volume_max).all()
        && subtracted_volume_min.cmple(volume_min).all()
    {
        return vec![];
    }

    // Compute the intersection of the two AABBs
    let overlap_min = volume_min.max(subtracted_volume_min);
    let overlap_max = volume_max.min(subtracted_volume_max);

    // If overlap volume is degenerated, return full area of first AABB
    if (overlap_max - overlap_min).min_element() <= 0 {
        return vec![Aabb3I::new(volume_min, volume_max)];
    }

    let mut result = Vec::with_capacity(6);

    // negative x volumne
    if volume_min.x < overlap_min.x {
        result.push(Aabb3I::new(
            volume_min,
            ivec3(overlap_min.x - 1, volume_max.y, volume_max.z),
        ));
    }

    // positive x volumne
    if overlap_max.x < volume_max.x {
        result.push(Aabb3I::new(
            ivec3(overlap_max.x + 1, volume_min.y, volume_min.z),
            volume_max,
        ));
    }

    // negative y volumne
    if volume_min.y < overlap_min.y {
        result.push(Aabb3I::new(
            ivec3(overlap_min.x, volume_min.y, volume_min.z),
            ivec3(overlap_max.x, overlap_min.y - 1, volume_max.z),
        ));
    }

    // positive y volumne
    if overlap_max.y < volume_max.y {
        result.push(Aabb3I::new(
            ivec3(overlap_min.x, overlap_max.y + 1, volume_min.z),
            ivec3(overlap_max.x, volume_max.y, volume_max.z),
        ));
    }

    // negative z volume
    if volume_min.z < overlap_min.z {
        result.push(Aabb3I::new(
            ivec3(overlap_min.x, overlap_min.y, volume_min.z),
            ivec3(overlap_max.x, overlap_max.y, overlap_min.z - 1),
        ));
    }

    // positive z volume
    if overlap_max.z < volume_max.z {
        result.push(Aabb3I::new(
            ivec3(overlap_min.x, overlap_min.y, overlap_max.z + 1),
            ivec3(overlap_max.x, overlap_max.y, volume_max.z),
        ));
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::tests::cmp_vec_unordered;

    use super::*;

    #[test]
    fn test_equivalent_areas() {
        assert_eq!(
            area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 0), ivec2(2, 2)),
                Aabb2I::new(ivec2(0, 0), ivec2(2, 2))
            ),
            vec![],
        );
    }

    #[test]
    fn test_zero_size() {
        assert_eq!(
            area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 0), ivec2(0, 0)),
                Aabb2I::new(ivec2(0, 0), ivec2(0, 0))
            ),
            vec![]
        );
    }

    #[test]
    fn test_partial_overlap_center() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 0), ivec2(3, 3)),
                Aabb2I::new(ivec2(1, 1), ivec2(2, 2)),
            ),
            &vec![
                Aabb2I::new(ivec2(0, 0), ivec2(0, 3)),
                Aabb2I::new(ivec2(3, 0), ivec2(3, 3)),
                Aabb2I::new(ivec2(1, 0), ivec2(2, 0)),
                Aabb2I::new(ivec2(1, 3), ivec2(2, 3)),
            ],
        )
    }

    #[test]
    fn test_fully_contained_area() {
        assert_eq!(
            area_subtract_overlap_2d(
                Aabb2I::new(ivec2(2, 2), ivec2(3, 3)),
                Aabb2I::new(ivec2(1, 1), ivec2(4, 4))
            ),
            vec![]
        );
    }

    #[test]
    fn test_overlap_x() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 1), ivec2(4, 2)),
                Aabb2I::new(ivec2(1, 0), ivec2(3, 3)),
            ),
            &vec![
                Aabb2I::new(ivec2(0, 1), ivec2(0, 2)),
                Aabb2I::new(ivec2(4, 1), ivec2(4, 2)),
            ],
        )
    }

    #[test]
    fn test_overlap_y() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_subtract_overlap_2d(
                Aabb2I::new(ivec2(1, 0), ivec2(3, 3)),
                Aabb2I::new(ivec2(0, 1), ivec2(4, 2)),
            ),
            &vec![
                Aabb2I::new(ivec2(1, 0), ivec2(3, 0)),
                Aabb2I::new(ivec2(1, 3), ivec2(3, 3)),
            ],
        )
    }

    #[test]
    fn test_half_overlap_y_positive() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 1), ivec2(4, 3)),
                Aabb2I::new(ivec2(1, 0), ivec2(2, 2)),
            ),
            &vec![
                Aabb2I::new(ivec2(0, 1), ivec2(0, 3)),
                Aabb2I::new(ivec2(1, 3), ivec2(2, 3)),
                Aabb2I::new(ivec2(3, 1), ivec2(4, 3)),
            ],
        )
    }

    #[test]
    fn test_disjoint_areas() {
        assert_eq!(
            area_subtract_overlap_2d(
                Aabb2I::new(ivec2(0, 0), ivec2(2, 2)),
                Aabb2I::new(ivec2(3, 3), ivec2(4, 4))
            ),
            vec![Aabb2I::new(ivec2(0, 0), ivec2(2, 2))]
        );
    }

    #[test]
    fn test_3d_disjoint_volumes() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(11, 11, 11);
        let sub_max = ivec3(20, 20, 20);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(result, vec![Aabb3I::new(volume_min, volume_max)]);
    }

    #[test]
    fn test_3d_equivalent_volumes() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(0, 0, 0);
        let sub_max = ivec3(10, 10, 10);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_3d_partial_overlap_center() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(3, 3, 3);
        let sub_max = ivec3(7, 7, 7);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(
            result,
            vec![
                Aabb3I::new(ivec3(0, 0, 0), ivec3(2, 10, 10)),  // Left
                Aabb3I::new(ivec3(8, 0, 0), ivec3(10, 10, 10)), // Right
                Aabb3I::new(ivec3(3, 0, 0), ivec3(7, 2, 10)),   // Bottom
                Aabb3I::new(ivec3(3, 8, 0), ivec3(7, 10, 10)),  // Top
                Aabb3I::new(ivec3(3, 3, 0), ivec3(7, 7, 2)),    // Front
                Aabb3I::new(ivec3(3, 3, 8), ivec3(7, 7, 10))    // Back
            ]
        );
    }

    #[test]
    fn test_3d_partial_overlap_edge() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 5, 5);
        let sub_max = ivec3(15, 15, 15);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(
            result,
            vec![
                Aabb3I::new(ivec3(0, 0, 0), ivec3(4, 10, 10)), // Left
                Aabb3I::new(ivec3(5, 0, 0), ivec3(10, 4, 10)), // Bottom
                Aabb3I::new(ivec3(5, 5, 0), ivec3(10, 10, 4))  // Front
            ]
        );
    }

    #[test]
    fn test_3d_no_volume_first_aabb() {
        let volume_min = ivec3(5, 5, 5);
        let volume_max = ivec3(5, 5, 5);
        let sub_min = ivec3(3, 3, 3);
        let sub_max = ivec3(7, 7, 7);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_3d_no_volume_second_aabb() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 5, 5);
        let sub_max = ivec3(5, 5, 5);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(result, vec![Aabb3I::new(volume_min, volume_max)]);
    }

    #[test]
    fn test_3d_overlap_only_one_axis() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 0, 0);
        let sub_max = ivec3(15, 10, 10);

        let result = volume_subtract_overlap_3d(
            Aabb3I::new(volume_min, volume_max),
            Aabb3I::new(sub_min, sub_max),
        );

        assert_eq!(
            result,
            vec![
                Aabb3I::new(ivec3(0, 0, 0), ivec3(4, 10, 10)), // Left
            ]
        );
    }
}
