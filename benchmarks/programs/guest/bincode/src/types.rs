extern crate alloc;

use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use bincode::{Decode, Encode};
use rand::Rng;

use crate::generate::{generate_vec, Generate};

#[derive(Encode, Decode, Debug, PartialEq)]
pub enum GameType {
    Survival,
    Creative,
    Adventure,
    Spectator,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Item {
    pub count: i8,
    pub slot: u8,
    pub id: String,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Abilities {
    pub walk_speed: f32,
    pub fly_speed: f32,
    pub may_fly: bool,
    pub flying: bool,
    pub invulnerable: bool,
    pub may_build: bool,
    pub instabuild: bool,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Motion {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Rotation {
    pub pitch: f32, // often refers to rotation around the X-axis
    pub yaw: f32,   // often refers to rotation around the Y-axis
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Entity {
    pub id: String,
    pub pos: Position,
    pub motion: Motion,
    pub rotation: Rotation,
    pub fall_distance: f32,
    pub fire: u16,
    pub air: u16,
    pub on_ground: bool,
    pub no_gravity: bool,
    pub invulnerable: bool,
    pub portal_cooldown: i32,
    pub uuid: [u32; 4],
    pub custom_name: Option<String>,
    pub custom_name_visible: bool,
    pub silent: bool,
    pub glowing: bool,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct RecipeBook {
    pub recipes: Vec<String>,
    pub to_be_displayed: Vec<String>,
    pub is_filtering_craftable: bool,
    pub is_gui_open: bool,
    pub is_furnace_filtering_craftable: bool,
    pub is_furnace_gui_open: bool,
    pub is_blasting_furnace_filtering_craftable: bool,
    pub is_blasting_furnace_gui_open: bool,
    pub is_smoker_filtering_craftable: bool,
    pub is_smoker_gui_open: bool,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Player {
    pub game_type: GameType,
    pub previous_game_type: GameType,
    pub score: i64,
    pub dimension: String,
    pub selected_item_slot: u32,
    pub selected_item: Item,
    pub spawn_dimension: Option<String>,
    pub spawn_x: i64,
    pub spawn_y: i64,
    pub spawn_z: i64,
    pub spawn_forced: Option<bool>,
    pub sleep_timer: u16,
    pub food_exhaustion_level: f32,
    pub food_saturation_level: f32,
    pub food_tick_timer: u32,
    pub xp_level: u32,
    pub xp_p: f32,
    pub xp_total: i32,
    pub xp_seed: i32,
    pub inventory: Vec<Item>,
    pub ender_items: Vec<Item>,
    pub abilities: Abilities,
    pub entered_nether_position: Option<Position>,
    pub shoulder_entity_left: Option<Entity>,
    pub shoulder_entity_right: Option<Entity>,
    pub seen_credits: bool,
    pub recipe_book: RecipeBook,
}

#[derive(Encode, Decode, Debug, PartialEq)]
pub struct Players {
    pub players: Vec<Player>,
}

impl Generate for GameType {
    fn generate<R: Rng>(rand: &mut R) -> Self {
        match rand.gen_range(0..4) {
            0 => GameType::Survival,
            1 => GameType::Creative,
            2 => GameType::Adventure,
            3 => GameType::Spectator,
            _ => unreachable!(),
        }
    }
}

impl Generate for Item {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        const IDS: [&str; 8] = [
            "dirt",
            "stone",
            "pickaxe",
            "sand",
            "gravel",
            "shovel",
            "chestplate",
            "steak",
        ];
        Self {
            count: rng.gen(),
            slot: rng.gen(),
            id: IDS[rng.gen_range(0..IDS.len())].to_string(),
        }
    }
}

impl Generate for Position {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        Self {
            x: rng.gen(),
            y: rng.gen(),
            z: rng.gen(),
        }
    }
}

impl Generate for Motion {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        Self {
            dx: rng.gen(),
            dy: rng.gen(),
            dz: rng.gen(),
        }
    }
}

impl Generate for Rotation {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        Self {
            pitch: rng.gen(),
            yaw: rng.gen(),
        }
    }
}

impl Generate for Entity {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        const IDS: [&str; 8] = [
            "cow", "sheep", "zombie", "skeleton", "spider", "creeper", "parrot", "bee",
        ];
        const CUSTOM_NAMES: [&str; 8] = [
            "rainbow", "princess", "steve", "johnny", "missy", "coward", "fairy", "howard",
        ];

        Self {
            id: IDS[rng.gen_range(0..IDS.len())].to_string(),
            pos: Position::generate(rng),
            motion: Motion::generate(rng),
            rotation: Rotation::generate(rng),
            fall_distance: rng.gen(),
            fire: rng.gen(),
            air: rng.gen(),
            on_ground: rng.gen_bool(0.5),
            no_gravity: rng.gen_bool(0.5),
            invulnerable: rng.gen_bool(0.5),
            portal_cooldown: rng.gen(),
            uuid: <[u32; 4] as Generate>::generate(rng),
            custom_name: <Option<()> as Generate>::generate(rng)
                .map(|_| CUSTOM_NAMES[rng.gen_range(0..CUSTOM_NAMES.len())].to_string()),
            custom_name_visible: rng.gen_bool(0.5),
            silent: rng.gen_bool(0.5),
            glowing: rng.gen_bool(0.5),
        }
    }
}

impl Generate for Abilities {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        Self {
            walk_speed: rng.gen(),
            fly_speed: rng.gen(),
            may_fly: rng.gen_bool(0.5),
            flying: rng.gen_bool(0.5),
            invulnerable: rng.gen_bool(0.5),
            may_build: rng.gen_bool(0.5),
            instabuild: rng.gen_bool(0.5),
        }
    }
}

impl Generate for RecipeBook {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        const RECIPES: [&str; 8] = [
            "pickaxe",
            "torch",
            "bow",
            "crafting table",
            "furnace",
            "shears",
            "arrow",
            "tnt",
        ];
        const MAX_RECIPES: usize = 30;
        const MAX_DISPLAYED_RECIPES: usize = 10;
        Self {
            recipes: generate_vec::<_, ()>(rng, 0..MAX_RECIPES)
                .iter()
                .map(|_| RECIPES[rng.gen_range(0..RECIPES.len())].to_string())
                .collect(),
            to_be_displayed: generate_vec::<_, ()>(rng, 0..MAX_DISPLAYED_RECIPES)
                .iter()
                .map(|_| RECIPES[rng.gen_range(0..RECIPES.len())].to_string())
                .collect(),
            is_filtering_craftable: rng.gen_bool(0.5),
            is_gui_open: rng.gen_bool(0.5),
            is_furnace_filtering_craftable: rng.gen_bool(0.5),
            is_furnace_gui_open: rng.gen_bool(0.5),
            is_blasting_furnace_filtering_craftable: rng.gen_bool(0.5),
            is_blasting_furnace_gui_open: rng.gen_bool(0.5),
            is_smoker_filtering_craftable: rng.gen_bool(0.5),
            is_smoker_gui_open: rng.gen_bool(0.5),
        }
    }
}

impl Generate for Player {
    fn generate<R: Rng>(rng: &mut R) -> Self {
        const DIMENSIONS: [&str; 3] = ["overworld", "nether", "end"];
        const MAX_ITEMS: usize = 40;
        const MAX_ENDER_ITEMS: usize = 27;
        Self {
            game_type: GameType::generate(rng),
            previous_game_type: GameType::generate(rng),
            score: rng.gen(),
            dimension: DIMENSIONS[rng.gen_range(0..DIMENSIONS.len())].to_string(),
            selected_item_slot: rng.gen(),
            selected_item: Item::generate(rng),
            spawn_dimension: <Option<()> as Generate>::generate(rng)
                .map(|_| DIMENSIONS[rng.gen_range(0..DIMENSIONS.len())].to_string()),
            spawn_x: rng.gen(),
            spawn_y: rng.gen(),
            spawn_z: rng.gen(),
            spawn_forced: <Option<bool> as Generate>::generate(rng),
            sleep_timer: rng.gen(),
            food_exhaustion_level: rng.gen(),
            food_saturation_level: rng.gen(),
            food_tick_timer: rng.gen(),
            xp_level: rng.gen(),
            xp_p: rng.gen(),
            xp_total: rng.gen(),
            xp_seed: rng.gen(),
            inventory: generate_vec(rng, 0..MAX_ITEMS),
            ender_items: generate_vec(rng, 0..MAX_ENDER_ITEMS),
            abilities: Abilities::generate(rng),
            entered_nether_position: <Option<Position> as Generate>::generate(rng),
            shoulder_entity_left: <Option<Entity> as Generate>::generate(rng),
            shoulder_entity_right: <Option<Entity> as Generate>::generate(rng),
            seen_credits: rng.gen_bool(0.5),
            recipe_book: RecipeBook::generate(rng),
        }
    }
}
