use glam::{ivec2, ivec3, IVec2, IVec3};

/// Returns a set of disjoint 2d volumes within the first AABB, that aren't overlapped by the second AABB.
///
/// If the second AABB isn't fully covering the first AABB, and either:
///
/// 1. AABB don't overlap at all
/// 2. Min vectors aren't lower than max vectors in every dimension
/// 3. Either AABB has one side of zero length
///
/// Then `vec![(area_min, area_max)]` is returned.
pub fn area_substract_overlap_2d(
    area_min: IVec2,
    area_max: IVec2,
    substracted_area_min: IVec2,
    substracted_area_max: IVec2,
) -> Vec<(IVec2, IVec2)> {
    // Early return if the area is entirely within the substracted area
    if substracted_area_max.cmpge(area_max).all() && substracted_area_min.cmple(area_min).all() {
        return vec![];
    }

    let overlap_min = area_min.max(substracted_area_min);
    let overlap_max = area_max.min(substracted_area_max);

    // If overlap area is degenerated, return full area of first AABB
    if (overlap_max - overlap_min).min_element() <= 0 {
        return vec![(area_min, area_max)];
    }

    let mut result = Vec::new();

    // negative x area
    if area_min.x < overlap_min.x {
        result.push((area_min, ivec2(overlap_min.x, area_max.y)));
    }

    // positive x area
    if area_max.x > overlap_max.x {
        result.push((ivec2(overlap_max.x, area_min.y), area_max));
    }

    // negative y area
    if area_min.y < overlap_min.y {
        result.push((
            ivec2(overlap_min.x, area_min.y),
            ivec2(overlap_max.x, overlap_min.y),
        ));
    }

    // positive y area
    if area_max.y > overlap_max.y {
        result.push((
            ivec2(overlap_min.x, overlap_max.y),
            ivec2(overlap_max.x, area_max.y),
        ));
    }

    result
}

/// Returns a set of disjoint 3d volumes within the first AABB, that aren't overlapped by the second AABB.
///
/// If the second AABB isn't fully covering the first AABB, and either:
///
/// 1. AABB don't overlap at all
/// 2. Min vectors aren't lower than max vectors in every dimension
/// 3. Either AABB has one side of zero length
///
/// Then `vec![(volume_min, volume_max)]` is returned.
pub fn volume_subtract_overlap_3d(
    volume_min: IVec3,
    volume_max: IVec3,
    substracted_volume_min: IVec3,
    substracted_volume_max: IVec3,
) -> Vec<(IVec3, IVec3)> {
    // Early return if the area is entirely within the substracted area
    if substracted_volume_max.cmpge(volume_max).all()
        && substracted_volume_min.cmple(volume_min).all()
    {
        return vec![];
    }

    // Compute the intersection of the two AABBs
    let overlap_min = volume_min.max(substracted_volume_min);
    let overlap_max = volume_max.min(substracted_volume_max);

    // If overlap volume is degenerated, return full area of first AABB
    if (overlap_max - overlap_min).min_element() <= 0 {
        return vec![(volume_min, volume_max)];
    }

    let mut result = Vec::new();

    // negative x volumne
    if volume_min.x < overlap_min.x {
        result.push((volume_min, ivec3(overlap_min.x, volume_max.y, volume_max.z)));
    }

    // positive x volumne
    if overlap_max.x < volume_max.x {
        result.push((ivec3(overlap_max.x, volume_min.y, volume_min.z), volume_max));
    }

    // negative y volumne
    if volume_min.y < overlap_min.y {
        result.push((
            ivec3(overlap_min.x, volume_min.y, volume_min.z),
            ivec3(overlap_max.x, overlap_min.y, volume_max.z),
        ));
    }

    // positive y volumne
    if overlap_max.y < volume_max.y {
        result.push((
            ivec3(overlap_min.x, overlap_max.y, volume_min.z),
            ivec3(overlap_max.x, volume_max.y, volume_max.z),
        ));
    }

    // negative z volume
    if volume_min.z < overlap_min.z {
        result.push((
            ivec3(overlap_min.x, overlap_min.y, volume_min.z),
            ivec3(overlap_max.x, overlap_max.y, overlap_min.z),
        ));
    }

    // positive z volume
    if overlap_max.z < volume_max.z {
        result.push((
            ivec3(overlap_min.x, overlap_min.y, overlap_max.z),
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
            area_substract_overlap_2d(ivec2(0, 0), ivec2(2, 2), ivec2(0, 0), ivec2(2, 2)),
            vec![],
        );
    }

    #[test]
    fn test_zero_size() {
        assert_eq!(
            area_substract_overlap_2d(ivec2(0, 0), ivec2(0, 0), ivec2(0, 0), ivec2(0, 0)),
            vec![]
        );
    }

    #[test]
    fn test_partial_overlap_center() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_substract_overlap_2d(ivec2(0, 0), ivec2(3, 3), ivec2(1, 1), ivec2(2, 2)),
            &vec![
                (ivec2(0, 0), ivec2(1, 3)),
                (ivec2(2, 0), ivec2(3, 3)),
                (ivec2(1, 0), ivec2(2, 1)),
                (ivec2(1, 2), ivec2(2, 3)),
            ],
        )
    }

    #[test]
    fn test_fully_contained_area() {
        assert_eq!(
            area_substract_overlap_2d(ivec2(2, 2), ivec2(3, 3), ivec2(1, 1), ivec2(4, 4)),
            vec![]
        );
    }

    #[test]
    fn test_overlap_x() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_substract_overlap_2d(ivec2(0, 1), ivec2(4, 2), ivec2(1, 0), ivec2(3, 3)),
            &vec![(ivec2(0, 1), ivec2(1, 2)), (ivec2(3, 1), ivec2(4, 2))],
        )
    }

    #[test]
    fn test_overlap_y() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_substract_overlap_2d(ivec2(1, 0), ivec2(3, 3), ivec2(0, 1), ivec2(4, 2)),
            &vec![(ivec2(1, 0), ivec2(3, 1)), (ivec2(1, 2), ivec2(3, 3))],
        )
    }

    #[test]
    fn test_half_overlap_y_positive() -> Result<(), ()> {
        cmp_vec_unordered(
            &area_substract_overlap_2d(ivec2(0, 1), ivec2(4, 3), ivec2(1, 0), ivec2(2, 2)),
            &vec![
                (ivec2(0, 1), ivec2(1, 3)),
                (ivec2(1, 2), ivec2(2, 3)),
                (ivec2(2, 1), ivec2(4, 3)),
            ],
        )
    }

    #[test]
    fn test_disjoint_areas() {
        assert_eq!(
            area_substract_overlap_2d(ivec2(0, 0), ivec2(2, 2), ivec2(3, 3), ivec2(4, 4)),
            vec![(ivec2(0, 0), ivec2(2, 2))]
        );
    }

    #[test]
    fn test_3d_disjoint_volumes() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(11, 11, 11);
        let sub_max = ivec3(20, 20, 20);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(result, vec![(volume_min, volume_max)]);
    }

    #[test]
    fn test_3d_equivalent_volumes() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(0, 0, 0);
        let sub_max = ivec3(10, 10, 10);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_3d_partial_overlap_center() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(3, 3, 3);
        let sub_max = ivec3(7, 7, 7);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(
            result,
            vec![
                (ivec3(0, 0, 0), ivec3(3, 10, 10)),  // Left
                (ivec3(7, 0, 0), ivec3(10, 10, 10)), // Right
                (ivec3(3, 0, 0), ivec3(7, 3, 10)),   // Bottom
                (ivec3(3, 7, 0), ivec3(7, 10, 10)),  // Top
                (ivec3(3, 3, 0), ivec3(7, 7, 3)),    // Front
                (ivec3(3, 3, 7), ivec3(7, 7, 10))    // Back
            ]
        );
    }

    #[test]
    fn test_3d_partial_overlap_edge() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 5, 5);
        let sub_max = ivec3(15, 15, 15);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(
            result,
            vec![
                (ivec3(0, 0, 0), ivec3(5, 10, 10)), // Left
                (ivec3(5, 0, 0), ivec3(10, 5, 10)), // Bottom
                (ivec3(5, 5, 0), ivec3(10, 10, 5))  // Front
            ]
        );
    }

    #[test]
    fn test_3d_no_volume_first_aabb() {
        let volume_min = ivec3(5, 5, 5);
        let volume_max = ivec3(5, 5, 5);
        let sub_min = ivec3(3, 3, 3);
        let sub_max = ivec3(7, 7, 7);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_3d_no_volume_second_aabb() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 5, 5);
        let sub_max = ivec3(5, 5, 5);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(result, vec![(volume_min, volume_max)]);
    }

    #[test]
    fn test_3d_overlap_only_one_axis() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 0, 0);
        let sub_max = ivec3(15, 10, 10);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(
            result,
            vec![
                (ivec3(0, 0, 0), ivec3(5, 10, 10)), // Left
            ]
        );
    }

    #[test]
    fn test_3d_overlap_corner() {
        let volume_min = ivec3(0, 0, 0);
        let volume_max = ivec3(10, 10, 10);
        let sub_min = ivec3(5, 5, 5);
        let sub_max = ivec3(15, 15, 15);

        let result = volume_subtract_overlap_3d(volume_min, volume_max, sub_min, sub_max);

        assert_eq!(
            result,
            vec![
                (ivec3(0, 0, 0), ivec3(5, 10, 10)), // Left
                (ivec3(5, 0, 0), ivec3(10, 5, 10)), // Bottom
                (ivec3(5, 5, 0), ivec3(10, 10, 5)), // Front
            ]
        );
    }
}
