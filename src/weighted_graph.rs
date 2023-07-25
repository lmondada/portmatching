use petgraph::visit::{GraphBase, GraphRef, IntoNodeIdentifiers};
use portgraph::PortView;

#[derive(Debug, Copy, Clone)]
pub struct WeightedGraphRef<G, W> {
    graph: G,
    weights: W,
}

impl<G, W> WeightedGraphRef<G, W> {
    pub fn new(graph: G, weights: W) -> Self {
        Self { graph, weights }
    }
}

impl<G, W> From<(G, W)> for WeightedGraphRef<G, W> {
    fn from((graph, weights): (G, W)) -> Self {
        Self::new(graph, weights)
    }
}

impl<G, W> From<WeightedGraphRef<G, W>> for (G, W) {
    fn from(graph: WeightedGraphRef<G, W>) -> Self {
        (graph.graph, graph.weights)
    }
}

impl<G: PortView, W> PortView for WeightedGraphRef<G, W> {
    type Nodes<'a> = G::Nodes<'a>
    where
        Self: 'a;

    type Ports<'a> = G::Ports<'a>
    where
        Self: 'a;

    type NodePorts<'a> = G::NodePorts<'a>
    where
        Self: 'a;

    type NodePortOffsets<'a> = G::NodePortOffsets<'a>
    where
        Self: 'a;

    fn port_direction(
        &self,
        port: impl Into<portgraph::PortIndex>,
    ) -> Option<portgraph::Direction> {
        self.graph.port_direction(port)
    }

    fn port_node(&self, port: impl Into<portgraph::PortIndex>) -> Option<portgraph::NodeIndex> {
        self.graph.port_node(port)
    }

    fn port_offset(&self, port: impl Into<portgraph::PortIndex>) -> Option<portgraph::PortOffset> {
        self.graph.port_offset(port)
    }

    fn port_index(
        &self,
        node: portgraph::NodeIndex,
        offset: portgraph::PortOffset,
    ) -> Option<portgraph::PortIndex> {
        self.graph.port_index(node, offset)
    }

    fn ports(
        &self,
        node: portgraph::NodeIndex,
        direction: portgraph::Direction,
    ) -> Self::NodePorts<'_> {
        self.graph.ports(node, direction)
    }

    fn all_ports(&self, node: portgraph::NodeIndex) -> Self::NodePorts<'_> {
        self.graph.all_ports(node)
    }

    fn input(&self, node: portgraph::NodeIndex, offset: usize) -> Option<portgraph::PortIndex> {
        self.graph.input(node, offset)
    }

    fn output(&self, node: portgraph::NodeIndex, offset: usize) -> Option<portgraph::PortIndex> {
        self.graph.output(node, offset)
    }

    fn num_ports(&self, node: portgraph::NodeIndex, direction: portgraph::Direction) -> usize {
        self.graph.num_ports(node, direction)
    }

    fn port_offsets(
        &self,
        node: portgraph::NodeIndex,
        direction: portgraph::Direction,
    ) -> Self::NodePortOffsets<'_> {
        self.graph.port_offsets(node, direction)
    }

    fn all_port_offsets(&self, node: portgraph::NodeIndex) -> Self::NodePortOffsets<'_> {
        self.graph.all_port_offsets(node)
    }

    fn contains_node(&self, node: portgraph::NodeIndex) -> bool {
        self.graph.contains_node(node)
    }

    fn contains_port(&self, port: portgraph::PortIndex) -> bool {
        self.graph.contains_port(port)
    }

    fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn port_count(&self) -> usize {
        self.graph.port_count()
    }

    fn nodes_iter(&self) -> Self::Nodes<'_> {
        self.graph.nodes_iter()
    }

    fn ports_iter(&self) -> Self::Ports<'_> {
        self.graph.ports_iter()
    }

    fn node_capacity(&self) -> usize {
        self.graph.node_capacity()
    }

    fn port_capacity(&self) -> usize {
        self.graph.port_capacity()
    }

    fn node_port_capacity(&self, node: portgraph::NodeIndex) -> usize {
        self.graph.node_port_capacity(node)
    }
}

impl<G: GraphBase, W> GraphBase for WeightedGraphRef<G, W> {
    type EdgeId = G::EdgeId;
    type NodeId = G::NodeId;
}

impl<G: GraphRef + Copy, W: Copy> GraphRef for WeightedGraphRef<G, W> {}

impl<G: IntoNodeIdentifiers, W: Copy> IntoNodeIdentifiers for WeightedGraphRef<G, W> {
    type NodeIdentifiers = <G as IntoNodeIdentifiers>::NodeIdentifiers;

    fn node_identifiers(self) -> Self::NodeIdentifiers {
        self.graph.node_identifiers()
    }
}
