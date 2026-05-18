from __future__ import annotations

import contextlib
from functools import lru_cache
from io import StringIO
import logging
from constructive_geometries import Geomatcher
from .utils import (
    load_builtin_topologies,
    load_legacy_geographies,
    load_missing_geographies,
    get_str,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@contextlib.contextmanager
def _silent_geomatcher_lookup():
    """Suppress stdout/stderr and country_converter log chatter during probes."""
    logger_names = ("country_converter", "country_converter.country_converter")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous = [(log.disabled, log.level) for log in loggers]
    for log in loggers:
        log.disabled = True
    try:
        with (
            contextlib.redirect_stdout(StringIO()),
            contextlib.redirect_stderr(StringIO()),
        ):
            yield
    finally:
        for log, (disabled, level) in zip(loggers, previous):
            log.disabled = disabled
            log.setLevel(level)


class GeoResolver:
    """
    Resolve geographic containment/coverage using constructive_geometries + project weights.

    :param weights: Mapping of (supplier_loc, consumer_loc) tuples to numeric weights.
    :return: GeoResolver instance.
    """

    def __init__(
        self,
        weights: dict,
        additional_topologies: dict = None,
        use_builtin_topologies: bool = True,
    ):
        """
        Initialize the resolver and normalize internal weight keys.

        :param weights: Mapping of (supplier_loc, consumer_loc) -> weight value.
        :return: None
        """
        # Keep supplier/consumer keys intact when provided as tuples.
        # Backward-compatible: also accept flat string keys.
        norm_weights = {}
        for k, v in weights.items():
            if isinstance(k, tuple) and len(k) == 2:
                norm_key = (get_str(k[0]), get_str(k[1]))
            else:
                # Legacy flat location key; keep on both sides
                loc = get_str(k)
                norm_key = (loc, loc)
            norm_weights[norm_key] = v

        self.weights = norm_weights
        self.available_supplier_locations = {s for s, _ in self.weights.keys()}
        self.available_consumer_locations = {c for _, c in self.weights.keys()}
        self.available_locations = (
            self.available_supplier_locations | self.available_consumer_locations
        )
        self.weights_key = ",".join(sorted(f"{s}|{c}" for s, c in self.weights.keys()))
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Dependencies from constructive_geometries and your utils
        self.geo = Geomatcher()
        self.missing_geographies = load_missing_geographies()
        self.legacy_geographies = load_legacy_geographies()

        if use_builtin_topologies:
            for namespace, topology in load_builtin_topologies().items():
                self._add_topology_definitions(topology, namespace)
        self.contructive_geometry_namespaces = []

        if additional_topologies:
            basin_intersections = additional_topologies["basin_topologies"]
            self._add_topology_definitions({key:value for key,value in additional_topologies.items() if key !="basin_topologies"}, "ecoinvent")
        else:
            basin_intersections = None
        self._add_topology_definitions({"World": ["GLO", "RoW"]}, "ecoinvent")

        # allow specification of basins 
        if any(["basin_" in x for x in self.available_locations]):
            #print("LCIA method contains basin locations")
            self.logger.info("LCIA method contains basin locations")
        else:
            self.logger.warning("LCIA method contains no basin locations")
        if basin_intersections is None:
            self.logger.warning("couldn't find basin information in additional_topologies")
        else:
            basin_topologies = self._split_faces_by_basins(self.geo, basin_intersections)
            self.geo.add_definitions(basin_topologies, "AWARE", relative=False)
            self.logger.info("added basin geometries to georesolver")
            self.contructive_geometry_namespaces.append("AWARE")

    def _normalize_location(self, location: str) -> str | None:
        """Normalize noisy legacy labels before consulting Geomatcher."""
        cleaned = " ".join(get_str(location).split()).strip().rstrip(", ")
        unresolvable = set(
            self.legacy_geographies.get("unresolvable_placeholders", []) or []
        )
        if cleaned in unresolvable:
            return None
        aliases = self.legacy_geographies.get("aliases", {}) or {}
        return aliases.get(cleaned, cleaned)

    def _clean_topology_definitions(
        self, definitions: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Apply Edges geography aliases to topology members before registration."""
        cleaned = {}
        for region, members in (definitions or {}).items():
            region_key = " ".join(get_str(region).split()).strip().rstrip(", ")
            cleaned_members = []
            for member in members or []:
                normalized = self._normalize_location(member)
                if normalized is not None:
                    cleaned_members.append(normalized)
            cleaned[region_key] = cleaned_members
        return cleaned

    def _add_topology_definitions(
        self, definitions: dict[str, list[str]], namespace: str
    ) -> None:
        """Register topology definitions without leaking lookup diagnostics."""
        cleaned = self._clean_topology_definitions(definitions)
        with _silent_geomatcher_lookup():
            try:
                self.geo.add_definitions(cleaned, namespace, relative=True)
            except KeyError as exc:
                self.logger.info(
                    "Skipping topology namespace %s because a member could not be resolved: %s",
                    namespace,
                    exc,
                )

    def _resolve_geomatcher_keys(self, location: str) -> tuple[str | tuple, ...]:
        """
        Resolve all Geomatcher keys that can represent a location string.

        constructive_geometries delegates unknown string handling to
        country_converter, which writes "not found" messages to stderr and ISO3
        fallbacks to stdout before raising or returning. Edges treats these as
        normal failed fallback candidates, so keep them out of notebook output.
        """
        keys = []

        with _silent_geomatcher_lookup():
            try:
                keys.append(self.geo._actual_key(location))
            except KeyError:
                pass

        for key in self.geo.topology:
            if isinstance(key, tuple) and len(key) >= 2 and get_str(key) == location:
                keys.append(key)

        unique = []
        seen = set()
        for key in keys:
            if key not in seen:
                unique.append(key)
                seen.add(key)
        return tuple(unique)
        
    def possible_locations_from_constr_geom(self, constr_geom_method, location, exclusive):
        """
        Return related locations via constructive_geometries.

        This is a low-level helper used by `find_locations()`.
        It queries `self.geo.<constr_geom_method>()` and normalizes the result
        to string location codes via `get_str()`.

        Parameters
        ----------
        constr_geom_method : str
            Constructive-geometries method name; expected values:
            - ``"within"``: return things that contain `location`
            - ``"contained"``: return things contained by `location`
        location : str
            Base location identifier (e.g. ISO code, ecoinvent region ID,
            or custom namespace location).
        exclusive : bool, optional
            Whether to exclude the base location region itself from matches
            (passed through to `constructive_geometries`).

        Returns
        -------
        list[str]
            Size-sorted raw candidate locations (stringified) returned by the
            constructive_geometries query. 
        """
        raw_candidates = []
        # apply either the constructive_geometries funtion:
        #  within() => all locations that contain the provided location
        #  contained() => all locations that are inside the provided location
        for e in getattr(self.geo, constr_geom_method)(
            location,
            biggest_first=False, # ensures the results are sorted according to "size" (number of constructive geometry faces)
            exclusive=exclusive,
            include_self=False,
        ):
            # getattr(self.geo, constr_geom_method) is a list of constructive_geometries locations. 
            # it will include locations that are defined as tuple, e.g. ('ecoinvent', "UN-AMERICAS") 
            # the main "currency" of this package is the geography name as provided in the LCI database.
            # So, for matching to brighway edges we do not need the first entry of the tuple.
            raw_candidates.append(get_str(e))
        return raw_candidates


    def find_locations(
        self,
        location: str,
        weights_available: tuple,
        containing: bool = True,
        exceptions: tuple | None = None,
    ) -> list[str]:
        """
        Find locations that contain (or are contained by) a given location, filtered by availability.

        :param location: Base location code to resolve from. If the location code is provided as a tuple, the second entry of the tuple is used as location instead
        :param weights_available: Iterable of allowed region codes to consider.
        :param containing: If True, return regions that contain the base location; else contained regions.
        :param exceptions: Optional tuple of region codes to exclude.
        :return: List of matching region codes, filtered and ordered as discovered.
        """
        results = []
        print(f"running find_locations for {location}")
        # in this function we have the issue that basin is not found and no other CF is applied instead
        if exceptions:
            exceptions = tuple(get_str(e) for e in exceptions)

        original_location = get_str(location)
        location = self._normalize_location(original_location)
        if location is None:
            return results

        if (
            location != original_location
            and location in weights_available
            and (not exceptions or location not in exceptions)
        ):
            results.append(location)

        if location in self.missing_geographies:
            for e in self.missing_geographies[location]:
                e_str = get_str(e)
                if e_str in weights_available and e_str != location:
                    if not exceptions or e_str not in exceptions:
                        results.append(e_str)
        else:
            #check if location exists as such
            resolved_locations = self._resolve_geomatcher_keys(location)
            if not resolved_locations:
                self.logger.info("Region %s: no geometry found.", location)
                return sorted(set(results))

            #return linked locations
            method = "contained" if containing else "within"
            try:
                for resolved_location in resolved_locations:
                    raw_candidates = self.possible_locations_from_constr_geom(
                        constr_geom_method=method,
                        location=resolved_location,
                        exclusive=containing
                        )
                    for raw_cand in raw_candidates:
                        if (
                            raw_cand in weights_available
                            and raw_cand != location
                            and (not exceptions or raw_cand not in exceptions)
                        ):
                            results.append(raw_cand)
                            if not containing:
                                break # list is expected to start with GLO when within() is used in constructive_geometries. Only GLO is taken as return value.

            except KeyError:
                if len(self.contructive_geometry_namespaces):
                    # if self.geo has other namespaces added except 'ecoinvent' and their content is not found in country_converter
                    self.logger.info("Region %s: no geometry found, trying additional self.geo namespaces (fails are silent).", location)
                    for n in self.contructive_geometry_namespaces:
                        try:
                            raw_candidates = self.possible_locations_from_constr_geom(constr_geom_method=method,
                                                                          location=(n, location),
                                                                          exclusive=containing)
                            for raw_cand in raw_candidates:
                                print(raw_cand)
                                if (
                                    raw_cand in weights_available
                                    and raw_cand != location
                                    and (not exceptions or raw_cand not in exceptions)
                                ):
                                    results.append(raw_cand)
                                    print(results)
                                    if not containing:
                                        break # results list is expected to start with smallest geometry that contains the location.
                        except KeyError:
                            print(f"namespace {n} did not work out for {location}")
                            pass
                else:
                    self.logger.info("Region %s: no geometry found.", location)



        # Deduplicate and enforce deterministic ordering
        print("finished find_locations")
        return sorted(set(results))

    @lru_cache(maxsize=2048)
    def _cached_lookup(
        self, location: str, containing: bool, exceptions: tuple | None = None
    ) -> list:
        """
        Cached backend for resolving candidate locations.

        :param location: Base location code.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions: Optional tuple of region codes to exclude.
        :return: List of candidate region codes.
        """
        return self.find_locations(
            location=location,
            # GeoResolver resolves candidate geographies agnostic of side.
            # Side-specific filtering is handled in resolve_candidate_locations().
            weights_available=tuple(self.available_locations),
            containing=containing,
            exceptions=exceptions,
        )

    def resolve(
        self, location: str, containing=True, exceptions: list[str] | None = None
    ) -> list:
        """
        Resolve candidate regions for a given location with caching.

        :param location: Base location code.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions: Optional list of region codes to exclude.
        :return: List of candidate region codes.
        """
        return self._cached_lookup(
            location=get_str(location),
            containing=containing,
            exceptions=tuple(exceptions) if exceptions else None,
        )

    def batch(
        self,
        locations: list[str],
        containing=True,
        exceptions_map: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        """
        Resolve candidate regions for multiple locations at once.

        :param locations: List of base location codes.
        :param containing: If True, resolve containing regions; else contained regions.
        :param exceptions_map: Optional mapping of location -> list of regions to exclude.
        :return: Dict mapping each input location to its list of candidate region codes.
        """
        return {
            loc: self.resolve(
                loc, containing, exceptions_map.get(loc) if exceptions_map else None
            )
            for loc in locations
        }
    
    def _split_faces_by_basins(self,
        geomatcher: Geomatcher,
        basin_intersection: dict,
        allowed_misses: list[int] = None
        ) -> dict[str, set[int]]:
        """
        Split faces at basin boundaries and create basin topology definitions.
        
        Parameters:
        -----------
        geomatcher : Geomatcher
            The geomatcher object
        basin_intersection : dict
            maps face_id to basin_id(s)
        allowed_misses : list[int], optional
            Face IDs that do not need to be split, e.g. because they are not included in geomatcher even though existing in faces GeoPackage
            
        Returns:
        --------
        dict[str, set[int]]
            Basin topologies mapping basin IDs to face sets
        """
        from collections import defaultdict

        if allowed_misses is None:
            allowed_misses = [6893, 8281]  # Argentina-Chile conflict, Caspian Sea
        
        basin_topologies = defaultdict(set)
        max_face_int = max(x for x in geomatcher.faces if isinstance(x, int))
        lower_id_boundary = max_face_int + 1
        new_ids = [0]
        for face_id,basins in basin_intersection.items():
            face_id = int(face_id)
            if len(basins)==1:
                # face is contained in basin, does not need to be split
                basin_topologies[f"AWAREbas_{basins[0]}"].add(face_id)
                
            elif len(basins)>1:
                # face intersects at least two basins
                if face_id in allowed_misses:
                    logger.info(f"Skipping allowed missing basin: {face_id}")
                    continue
                # prescribing the new face_ids is significantly faster than just providing the number of new faces to geomatcher
                # geomatcher will willingly overwrite?? existing faces, so the ids need to be selected carefully.
                assert(max(new_ids)<lower_id_boundary)
                upper_id_boundary = lower_id_boundary + len(basins)
                new_ids = list(range(lower_id_boundary, upper_id_boundary))
                lower_id_boundary = upper_id_boundary
                
                geomatcher.split_face(face_id, ids=new_ids)
                logger.debug(f"Split face {face_id} into {len(new_ids)} parts")
                
                for i, basin in enumerate(basins):
                    basin_topologies[f"AWAREbas_{basin}"].add(new_ids[i])
            else:
                raise ValueError(f"No basin found for face_id {face_id}")
        
        return dict(basin_topologies)