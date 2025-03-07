use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    air::{
        air::{AirExpression, Bus, Constraint, Direction, Interaction},
        representation::RepresentationTable,
    },
    folder1::{
        file2_tree::ScopePath,
        file3::{Atom, FlatStatement, FlattenedFunction, Tree},
        ir::{Material, Statement, Type},
        stage1::Stage2Program,
        type_resolution::TypeSet,
    },
};

pub struct CellUsage {
    usages: Vec<ScopePath>,
}

#[derive(Default)]
pub struct AirConstructor {
    constraints: Vec<Constraint>,
    pub(crate) interactions: Vec<Interaction>,

    next_cell: usize,
    normal_cells: Vec<(usize, CellUsage)>,

    scope_expressions: HashMap<ScopePath, AirExpression>,
    reference_timestamp_expressions: HashMap<usize, AirExpression>,
    pub(crate) timestamp_before_after: HashMap<TimestampUsage, (AirExpression, usize)>,
    reference_multiplicity_cells: HashMap<usize, usize>,
}

impl AirConstructor {
    pub fn make_cell(&mut self) -> usize {
        let cell = self.next_cell;
        self.next_cell += 1;
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
        self.scope_expressions.insert(scope.clone(), expression);
    }

    pub fn add_before_after_timestamp(
        &mut self,
        usage: TimestampUsage,
        before_timestamp: AirExpression,
        after_timestamp: usize,
    ) {
        self.timestamp_before_after
            .insert(usage, (before_timestamp, after_timestamp));
    }

    pub fn new_multiplicity_cell(&mut self, statement_index: usize, scope: &ScopePath) -> usize {
        let cell = self.new_normal_cell(scope);
        self.reference_multiplicity_cells
            .insert(statement_index, cell);
        cell
    }
}

impl TypeSet {
    pub fn calc_type_size(&self, tipo: &Type) -> usize {
        match tipo {
            Type::Field => 1,
            Type::Reference(_) => 1,
            Type::Array(_) => 2,
            Type::UnderConstructionArray(_) => 2,
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
    num_timestamps: usize,
    timestamp_usages_here: Vec<TimestampUsage>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimestampUsage {
    FinalizeArray(usize),
    FunctionCall(usize),
}

impl AirTree {
    pub fn new(path_here: ScopePath) -> Self {
        Self {
            children: HashMap::new(),
            path_here,
            num_timestamps: 0,
            timestamp_usages_here: vec![],
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

    pub fn find_timestamp_usages(&mut self, function: &FlattenedFunction, program: &Stage2Program) {
        for (i, statement) in function.statements.iter().enumerate() {
            if statement.material == Material::Materialized {
                if let Statement::ArrayFinalization { .. } = statement.statement {
                    self.timestamp_usages_here
                        .push(TimestampUsage::FinalizeArray(i));
                }
            }
        }
        for (i, function_call) in function.function_calls.iter().enumerate() {
            if function_call.material == Material::Materialized
                && program.functions[&function_call.function_name].uses_timestamp
            {
                self.timestamp_usages_here
                    .push(TimestampUsage::FunctionCall(i));
            }
        }
        self.num_timestamps = self.timestamp_usages_here.len();
        for branch in self.children.values_mut() {
            let mut timestamps_here = 0;
            for child in branch.values_mut() {
                child.find_timestamp_usages(function, program);
                timestamps_here = timestamps_here.max(child.num_timestamps);
            }
            self.num_timestamps += timestamps_here;
        }
    }

    pub fn assign_before_after_timestamps(
        &self,
        mut before_timestamp: AirExpression,
        timestamps: &[usize],
        air_constructor: &mut AirConstructor,
    ) {
        let mut index = 0;
        for usage in self.timestamp_usages_here.iter() {
            air_constructor.add_before_after_timestamp(
                usage.clone(),
                before_timestamp.clone(),
                timestamps[index],
            );
            before_timestamp = AirExpression::single_cell(timestamps[index]);
            index += 1;
        }
        for branch in self.children.values() {
            let mut timestamps_here = 0;
            for child in branch.values() {
                timestamps_here = timestamps_here.max(child.num_timestamps);
            }
            if timestamps_here > 0 {
                let after_timestamp = timestamps[index + timestamps_here - 1];
                for child in branch.values() {
                    if child.num_timestamps == 0 {
                        air_constructor.add_scoped_constraint(
                            &child.path_here,
                            vec![before_timestamp.clone()],
                            &mut vec![Some(AirExpression::single_cell(after_timestamp))],
                        );
                    } else {
                        child.assign_before_after_timestamps(
                            before_timestamp.clone(),
                            &timestamps[index + timestamps_here - child.num_timestamps
                                ..index + timestamps_here],
                            air_constructor,
                        );
                    }
                }
                before_timestamp = AirExpression::single_cell(after_timestamp);
                index += timestamps_here;
            }
        }
    }
}

impl FlattenedFunction {
    fn construct_air(&self, program: &Stage2Program) -> AirConstructor {
        let mut relevant_scopes = vec![];

        for statement in self.statements.iter() {
            if statement.material == Material::Materialized {
                if let Statement::Equality { left: _, right } = &statement.statement {
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
            if matchi.material == Material::Materialized {
                for (constructor, _) in matchi.branches.iter() {
                    relevant_scopes.push(matchi.scope.then(matchi.index, constructor.clone()));
                }
            }
        }

        let mut air_tree = AirTree::new(ScopePath(vec![]));
        air_tree.init(&self.tree, &relevant_scopes);

        let mut air_constructor = AirConstructor::default();

        let (left_timestamp, start_timestamp) = if self.uses_timestamp {
            let left_timestamp = AirExpression::single_cell(air_constructor.make_cell());
            let mut amt = 0;
            for (i, statement) in self.statements.iter().enumerate() {
                if let FlatStatement {
                    material: Material::Materialized,
                    statement: Statement::Reference { .. },
                    ..
                } = statement
                {
                    air_constructor
                        .reference_timestamp_expressions
                        .insert(i, left_timestamp.plus(&AirExpression::constant(amt)));
                    amt += 1;
                }
            }
            (
                Some(left_timestamp.clone()),
                Some(left_timestamp.plus(&AirExpression::constant(amt))),
            )
        } else {
            (None, None)
        };

        air_tree.find_timestamp_usages(self, program);
        let right_timestamp = if air_tree.num_timestamps > 0 {
            let start_timestamp = start_timestamp.unwrap();
            let mut timestamps = vec![];
            for _ in 0..air_tree.num_timestamps {
                timestamps.push(air_constructor.make_cell());
            }
            air_tree.assign_before_after_timestamps(
                start_timestamp,
                &timestamps,
                &mut air_constructor,
            );
            Some(AirExpression::single_cell(*timestamps.last().unwrap()))
        } else {
            None
        };

        let mut representation_table = RepresentationTable::default();

        let mut function_call_interactions = vec![];
        for (i, function_call) in self.function_calls.iter().enumerate() {
            function_call_interactions.push(match function_call.material {
                Material::Materialized => {
                    let function = &program.functions[&function_call.function_name];
                    let mut fields = vec![AirExpression::constant(function.function_id as isize)];
                    if function.uses_timestamp {
                        let (before_timestamp, after_timestamp) = air_constructor
                            .timestamp_before_after[&TimestampUsage::FunctionCall(i)]
                            .clone();
                        fields.push(before_timestamp);
                        fields.push(AirExpression::single_cell(after_timestamp));
                    }
                    Some(Interaction {
                        bus: Bus::Function,
                        direction: Direction::Send,
                        multiplicity: AirExpression::one(),
                        fields,
                    })
                }
                Material::Dematerialized => None,
            });
        }

        let mut own_call_interaction = Interaction {
            bus: Bus::Function,
            direction: Direction::Receive,
            multiplicity: AirExpression::one(),
            fields: vec![AirExpression::constant(self.function_id as isize)],
        };
        if self.uses_timestamp {
            own_call_interaction.fields.push(left_timestamp.unwrap());
            own_call_interaction.fields.push(right_timestamp.unwrap());
        }

        for atoms in self.atoms_staged.iter() {
            for &atom in atoms {
                match atom {
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
                    Atom::InArgument(index) => {
                        let argument = &self.arguments[index];
                        let mut representation =
                            vec![None; program.types.calc_type_size(argument.get_type())];
                        argument.represent_top_down(
                            &program.types,
                            &mut representation_table,
                            &mut air_constructor,
                            &ScopePath::empty(),
                            &mut representation,
                        );
                        let representation: Vec<_> =
                            representation.into_iter().map(|x| x.unwrap()).collect();
                        own_call_interaction.fields.extend(representation);
                    }
                    Atom::OutArgument(index) => {
                        let argument = &self.arguments[index];
                        let mut representation =
                            vec![None; program.types.calc_type_size(argument.get_type())];
                        argument.represent_top_down(
                            &program.types,
                            &mut representation_table,
                            &mut air_constructor,
                            &ScopePath::empty(),
                            &mut representation,
                        );
                        let representation: Vec<_> =
                            representation.into_iter().map(|x| x.unwrap()).collect();
                        own_call_interaction.fields.extend(representation);
                    }
                }
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

        air_constructor.add_scoped_interaction(&ScopePath::empty(), own_call_interaction);

        air_constructor
    }
}
