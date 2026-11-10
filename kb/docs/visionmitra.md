# VisionMitra

A voice-first navigation and transit data platform. VisionMitra was built for
the way people actually move around Pune: on foot, on a bus, and often without
looking at a screen.

## Transit data

The system combines OpenStreetMap and Overpass data with Pune PMPML GTFS transit
data. Overpass supplies the street and point-of-interest graph; GTFS supplies
stops, routes and schedules. Joining the two is the core data problem, because
a bus stop in GTFS is a coordinate, not an address, and a pedestrian route ends
at a doorway, not a stop.

The pipeline processed more than thirteen hundred transit stops and more than
five hundred routes from the PMPML feed.

## Routing

Nearest-stop search uses a KD-tree over stop coordinates, which keeps the
lookup fast as the stop set grows. Route planning across the stop network uses
A*, with the heuristic tuned for transit rather than raw road distance. The
result is a plan that respects the stop graph instead of drawing a straight line
between two pins.

## Vision

A fine-tuned YOLOv8 model handles visual recognition. The intended use is
recognition of the things a transit user needs to identify — a stop, a vehicle,
a sign — so the voice layer can describe what is in front of the user.

## Interaction

The interface is voice-first. A user states a destination, the system resolves
the nearest stop, plans a route and speaks the next step. The design assumes the
screen may be in a pocket.

## Limitations

Data coverage is specific to Pune PMPML. The same architecture would need a
different feed, a different stop graph and a new evaluation for another city.
The routing and recognition work is a portfolio project, not an operated
navigation service, and there are no published field-accuracy numbers.

## What it shows

Joining two public datasets that were never designed to meet is a useful test of
engineering judgement. The interesting decisions were about identity — which
GTFS stop corresponds to which OSM node — not about the pathfinding algorithm.
