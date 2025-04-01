use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    air::{
        air::{AirExpression, Bus, Constraint, Direction, Interaction},
        representation::RepresentationTable,
    },
    core::{
        containers::Assertion,
        file3::{Atom, FlatStatement, FlattenedFunction, RepresentationOrder, Tree},
        ir::{Material, StatementVariant, Type},
        scope::ScopePath,
        stage1::Stage2Program,
        type_resolution::TypeSet,
    },
    parser::metadata::ParserMetadata,
};

pub struct CellUsage {
    usages: Vec<ScopePath>,
}

#[derive(Default)]
pub struct AirConstructor {
    constraints: Vec<Constraint>,
    pub(crate) interactions: Vec<Interaction>,
    row_index_cell: Option<usize>,

    next_cell: usize,
    normal_cells: Vec<(usize, CellUsage)>,

    scope_expressions: HashMap<ScopePath, AirExpression>,
    reference_address_expressions: HashMap<usize, AirExpression>,
    reference_multiplicity_cells: HashMap<usize, usize>,
}

impl AirConstructor {
    pub const MAX_HEIGHT: usize = 1 << 24;
    pub fn make_cell(&mut self) -> usize {
        let cell = self.next_cell;
        self.next_cell += 1;
        cell
    }

    pub fn have_row_index_cell(&mut self) -> usize {
        let cell = self.make_cell();
        self.row_index_cell = Some(cell);
        cell
    }

    pub fn new_normal_cell(&mut self, scope: &ScopePath) -> usize {
        let i = self
            .normal_cells
            .iter()
            .position(|usage| usage.1.usages.iter().all(|uscope| scope.disjoint(uscope)));
        if let Some(i) = i {
            self.normal_cells[i].1.usages.push(scope.clone());
            self.normal_cells[i].0
        } else {
            let cell = self.make_cell();
            self.normal_cells.push((
                cell,
                CellUsage {
                    usages: vec![scope.clone()],
                },
            ));
            cell
        }
    }

    pub fn add_single_constraint(&mut self, left: AirExpression, right: AirExpression) {
        self.constraints.push(Constraint { left, right });
    }

    pub fn get_scope_expression(&self, scope: &ScopePath) -> &AirExpression {
        &self.scope_expressions[scope]
    }

    pub fn add_scoped_constraint(
        &mut self,
        scope: &ScopePath,
        left: Vec<AirExpression>,
        right: &mut [Option<AirExpression>],
    ) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.into_iter().zip_eq(right.into_iter()) {
            if let Some(right) = right {
                let scope_expression = self.get_scope_expression(scope);
                let left = left.times(scope_expression);
                let right = right.times(scope_expression);
                self.add_single_constraint(left, right);
            } else {
                *right = Some(left);
            }
        }
    }

    pub fn add_scoped_single_constraint(
        &mut self,
        scope: &ScopePath,
        left: AirExpression,
        right: AirExpression,
    ) {
        self.add_scoped_constraint(scope, vec![left], &mut vec![Some(right)]);
    }

    pub fn add_scoped_interaction(&mut self, scope: &ScopePath, mut interaction: Interaction) {
        interaction.multiplicity = interaction
            .multiplicity
            .times(self.get_scope_expression(scope));
        self.interactions.push(interaction);
    }

    pub fn set_scope_expression(&mut self, scope: &ScopePath, expression: AirExpression) {
        self.add_single_constraint(expression.clone(), expression.times(&expression));
        self.scope_expressions.insert(scope.clone(), expression);
    }

    pub fn new_multiplicity_cell(&mut self, statement_index: usize, scope: &ScopePath) -> usize {
        let cell = self.new_normal_cell(scope);
        self.reference_multiplicity_cells
            .insert(statement_index, cell);
        cell
    }

    pub fn set_reference_address_expression(&mut self, index: usize, expression: AirExpression) {
        self.reference_address_expressions.insert(index, expression);
    }

    pub fn get_reference_address_expression(&mut self, index: usize) -> AirExpression {
        self.reference_address_expressions[&index].clone()
    }
}

impl TypeSet {
    pub fn calc_type_size(&self, tipo: &Type) -> usize {
        match tipo {
            Type::Field => 1,
            Type::Reference(_) => 1,
            Type::ReadablePrefix(..) => 1,
            Type::AppendablePrefix(..) => 1,
            Type::Array(_) => 1,
            Type::NamedType(name) => {
                let algebraic_type = &self.algebraic_types[name];
                let max_variant_size = algebraic_type
                    .variants
                    .iter()
                    .map(|variant| {
                        variant
                            .components
                            .iter()
                            .map(|component| self.calc_type_size(component))
                            .sum::<usize>()
                    })
                    .max()
                    .unwrap_or(0);
                max_variant_size
                    + if algebraic_type.variants.len() == 1 {
                        0
                    } else {
                        1
                    }
            }
            Type::Unmaterialized(_) => 0,
            Type::ConstArray(elem_type, len) => len * self.calc_type_size(elem_type),
        }
    }
}

pub struct AirTree {
    children: HashMap<usize, HashMap<String, AirTree>>,
    path_here: ScopePath,
}

impl AirTree {
    pub fn new(path_here: ScopePath) -> Self {
        Self {
            children: HashMap::new(),
            path_here,
        }
    }

    pub fn init(&mut self, tree: &Tree, relevant_scopes: &Vec<ScopePath>) {
        for (i, children) in tree.children.iter().enumerate() {
            if relevant_scopes
                .iter()
                .any(|scope| scope.is_prefix(&self.path_here, i))
            {
                let mut map = HashMap::new();
                for (name, node) in children {
                    let mut child = AirTree::new(self.path_here.then(i, name.clone()));
                    child.init(node, relevant_scopes);
                    map.insert(name.clone(), child);
                }
            }
        }
    }

    pub fn calc_scope_expressions(&self, air_constructor: &mut AirConstructor) -> AirExpression {
        let expression = if self.children.is_empty() {
            AirExpression::single_cell(air_constructor.make_cell())
        } else {
            let mut canonical_expression: Option<AirExpression> = None;
            for (_, children) in self.children.iter() {
                let mut this_expression = AirExpression::zero();
                for (_, child) in children {
                    this_expression =
                        this_expression.plus(&child.calc_scope_expressions(air_constructor));
                }
                if let Some(canonical_expression) = &canonical_expression {
                    air_constructor
                        .add_single_constraint(canonical_expression.clone(), this_expression);
                } else {
                    canonical_expression = Some(this_expression);
                }
            }

            canonical_expression.unwrap()
        };

        air_constructor.set_scope_expression(&self.path_here, expression.clone());
        expression
    }
}

impl FlattenedFunction {
    #[allow(dead_code)]
    fn construct_air(&self, program: &Stage2Program) -> AirConstructor {
        let mut relevant_scopes = vec![];

        for statement in self.statements.iter() {
            if statement.material == Material::Materialized {
                if let StatementVariant::Equality { left: _, right } = &statement.statement {
                    if program.types.calc_type_size(right.get_type()) > 0 {
                        relevant_scopes.push(statement.scope.clone());
                    }
                } else {
                    relevant_scopes.push(statement.scope.clone());
                }
            }
        }
        for function_call in self.function_calls.iter() {
            if function_call.material == Material::Materialized {
                relevant_scopes.push(function_call.scope.clone());
            }
        }
        for matchi in self.matches.iter() {
            if matchi.check_material == Material::Materialized {
                for branch in matchi.branches.iter() {
                    relevant_scopes
                        .push(matchi.scope.then(matchi.index, branch.constructor.clone()));
                }
            }
        }

        let mut air_tree = AirTree::new(ScopePath(vec![]));
        air_tree.init(&self.tree, &relevant_scopes);

        let mut air_constructor = AirConstructor::default();

        if self.creates_addresses {
            let mut offsets = HashMap::new();
            for (i, statement) in self.statements.iter().enumerate() {
                if let FlatStatement {
                    material: Material::Materialized,
                    statement: StatementVariant::Reference { .. },
                    ..
                } = statement
                {
                    offsets.insert(i, offsets.len());
                } else if let FlatStatement {
                    material: Material::Materialized,
                    statement: StatementVariant::EmptyPrefix { .. },
                    ..
                } = statement
                {
                    offsets.insert(i, offsets.len());
                }
            }
            let row_index_cell = AirExpression::single_cell(air_constructor.have_row_index_cell());
            let num_addresses = AirExpression::constant(offsets.len() as isize);
            let base =
                AirExpression::constant((self.function_id * AirConstructor::MAX_HEIGHT) as isize)
                    .plus(&row_index_cell.times(&num_addresses));
            for (i, offset) in offsets {
                air_constructor.set_reference_address_expression(
                    i,
                    base.plus(&AirExpression::constant(offset as isize)),
                );
            }
        }

        let mut representation_table = RepresentationTable::new(&self.declaration_set);

        let mut function_call_interactions = vec![];
        for function_call in self.function_calls.iter() {
            function_call_interactions.push(match function_call.material {
                Material::Materialized => {
                    let function = &program.functions[&function_call.function_name];
                    Some(Interaction {
                        bus: Bus::Function,
                        direction: Direction::Send,
                        multiplicity: AirExpression::one(),
                        fields: vec![AirExpression::constant(function.function_id as isize)],
                    })
                }
                Material::Dematerialized => None,
            });
        }

        for argument in self.arguments.iter() {
            if argument.represents {
                representation_table.fill_in_and_add_representation(
                    &mut air_constructor,
                    &ScopePath::empty(),
                    &argument.name,
                    &mut vec![None; program.types.calc_type_size(&argument.tipo)],
                );
            }
        }

        let representation_order = match &self.representation_order {
            RepresentationOrder::Inline(_) => unreachable!(),
            RepresentationOrder::NotInline(rep) => rep,
        };

        for atom in representation_order.iter() {
            match *atom {
                Atom::Statement(index) => {
                    self.statements[index].enforce(
                        index,
                        &program.types,
                        &mut representation_table,
                        &mut air_constructor,
                    );
                }
                Atom::Match(index) => {
                    self.matches[index].enforce(
                        &program.types,
                        &mut representation_table,
                        &mut air_constructor,
                    );
                }
                Atom::PartialFunctionCall(index, stage) => {
                    self.function_calls[index].represent_stage(
                        function_call_interactions
                            .get_mut(index)
                            .unwrap()
                            .as_mut()
                            .unwrap(),
                        stage,
                        &program.types,
                        &mut representation_table,
                        &mut air_constructor,
                    );
                }
            }
        }

        for Assertion {
            material,
            scope,
            left,
            right,
        } in self.assertions.iter()
        {
            if *material == Material::Materialized {
                air_constructor.add_scoped_constraint(
                    scope,
                    left.calc_representation(&program.types, &representation_table, scope),
                    &mut right
                        .calc_representation(&program.types, &representation_table, scope)
                        .into_iter()
                        .map(Some)
                        .collect_vec(),
                );
            }
        }

        for (i, statement) in self.statements.iter().enumerate() {
            if let FlatStatement {
                material: Material::Materialized,
                statement:
                    StatementVariant::PrefixAppend {
                        old_prefix,
                        elem,
                        new_prefix: _,
                    },
                scope,
                ..
            } = statement
            {
                let array_representation =
                    old_prefix.calc_representation(&program.types, &representation_table, scope);
                let elem_representation =
                    elem.calc_representation(&program.types, &representation_table, scope);
                let multiplicity_cell = air_constructor.new_multiplicity_cell(i, scope);
                let (_, length) = old_prefix
                    .get_type()
                    .get_appendable_prefix_type(Material::Materialized, &ParserMetadata::default())
                    .unwrap();
                let length_representation =
                    length.calc_representation(&program.types, &representation_table, scope);
                let mut fields = vec![];
                fields.extend(array_representation);
                fields.extend(length_representation);
                fields.extend(elem_representation);
                air_constructor.add_scoped_interaction(
                    scope,
                    Interaction {
                        bus: Bus::Array,
                        direction: Direction::Receive,
                        multiplicity: AirExpression::single_cell(multiplicity_cell),
                        fields,
                    },
                );
            }
        }

        for (function_call, interaction) in self
            .function_calls
            .iter()
            .zip_eq(function_call_interactions.into_iter())
        {
            if let Some(interaction) = interaction {
                air_constructor.add_scoped_interaction(&function_call.scope, interaction);
            }
        }

        let argument_representations = self.arguments.iter().flat_map(|argument| {
            representation_table.get_representation(&ScopePath::empty(), &argument.name)
        });
        let own_call_interaction = Interaction {
            bus: Bus::Function,
            direction: Direction::Receive,
            multiplicity: AirExpression::one(),
            fields: vec![AirExpression::constant(self.function_id as isize)]
                .into_iter()
                .chain(argument_representations)
                .collect(),
        };
        air_constructor.add_scoped_interaction(&ScopePath::empty(), own_call_interaction);

        air_constructor
    }
}
